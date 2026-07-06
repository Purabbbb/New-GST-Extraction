"""
json_exporter.py
=================
Single source of truth for serialising invoice extraction pipeline results
into a clean, human-readable JSON file.

This module performs NO extraction itself. It only consumes the outputs
already produced elsewhere in the pipeline:

    - header_extractor.extract_header()      -> OCR text / crop info
    - field_resolver.resolve_fields()         -> structured header fields
    - table_extractor.extract_tables()        -> list of table dataframes
    - qr_validator.validate_and_merge()        -> QR cross-validation report

and assembles them into one JSON-serialisable dict, then writes it to disk.

Usage
-----
    from json_exporter import export_invoice_json

    json_path = export_invoice_json(
        image_path=IMAGE_PATH,
        invoice=invoice,          # dict from resolve_fields() + qr_validator
        tables=tables,            # list from extract_tables()
        metadata={
            "model_used": "Newend.pt",
            "ocr_engine": "tesseract+easyocr",
        },
        processing_time=elapsed_seconds,
        qr_report=qr_report,      # dict from qr_validator.validate_and_merge()
    )
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "outputs_new"


# ---------------------------------------------------------------------------
# Value cleaning helpers
# ---------------------------------------------------------------------------

def _clean_value(value: Any) -> Any:
    """
    Normalise a single scalar value for JSON export.

    - None stays None.
    - Empty / whitespace-only strings -> None.
    - NaN / NaT (pandas or float) -> None.
    - Everything else is returned unchanged (strings are stripped).
    """
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def _clean_dict(d: dict) -> dict:
    """Apply `_clean_value` to every value in a flat dict."""
    return {k: _clean_value(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_header_section(invoice: Optional[dict]) -> dict:
    """
    Build the "header" section from the dict returned by
    field_resolver.resolve_fields() (as further enriched by qr_validator).

    Fields the resolver already produces are mapped directly. Fields the
    pipeline does not extract yet (PO number, bank details, etc.) are still
    included, set to null, so the JSON schema stays stable as extraction
    coverage grows -- consumers of the JSON never need to branch on
    whether a key exists.

    Note: by the time this runs in the full pipeline, qr_validator's
    ensure_no_nulls() has typically already replaced missing core fields
    with an explicit placeholder, so most of these will not actually be
    null in practice -- but _clean_value is still applied defensively.
    """
    invoice = invoice or {}

    return {
        "invoice_number": _clean_value(invoice.get("invoice_no")),
        "invoice_date": _clean_value(invoice.get("invoice_date")),
        "vendor_name": _clean_value(invoice.get("vendor_name")),
        "vendor_gstin": _clean_value(invoice.get("supplier_gstin")),
        "buyer_name": _clean_value(invoice.get("buyer_name")),
        "buyer_gstin": _clean_value(invoice.get("buyer_gstin")),
        "irn": _clean_value(invoice.get("irn")),
        "currency": _clean_value(invoice.get("currency")),
        "po_number": _clean_value(invoice.get("po_number")),
        "payment_terms": _clean_value(invoice.get("payment_terms")),
        "total_amount": _clean_value(invoice.get("total_amount")),
        "subtotal": _clean_value(invoice.get("subtotal")),
        "cgst": _clean_value(invoice.get("cgst")),
        "sgst": _clean_value(invoice.get("sgst")),
        "igst": _clean_value(invoice.get("igst")),
        "discount": _clean_value(invoice.get("discount")),
        "bank_details": invoice.get("bank_details") or {},
        "gstins_found": invoice.get("gstins_found") or [],
    }


def _build_tables_section(tables: Optional[list[dict]]) -> list[dict]:
    """
    Build the "tables" section from the list of dicts returned by
    table_extractor.extract_tables().

    Each input table dict is expected to have:
        table_number : int
        crop_path    : str
        dataframe    : pandas.DataFrame
    """
    if not tables:
        return []

    exported_tables: list[dict] = []

    for table in tables:
        df = table.get("dataframe")

        if df is None or df.empty:
            headers: list[str] = []
            rows: list[dict] = []
        else:
            headers = [str(c) for c in df.columns]
            rows = [_clean_dict(row) for row in df.to_dict(orient="records")]

        exported_tables.append(
            {
                "table_id": table.get("table_number"),
                "crop_path": table.get("crop_path"),
                "headers": headers,
                "rows": rows,
            }
        )

    return exported_tables


def _build_qr_validation_section(qr_report: Optional[dict]) -> dict:
    """
    Build the "qr_validation" section from the report returned by
    qr_validator.validate_and_merge().

    Always present in the output (even if no QR step ran) so consumers
    never have to branch on key existence:

        {
            "qr_status": "success" | "failed" | "not_run",
            "qr_reason": None | "no_qr_detected" | ...,
            "qr_confidence": 0.91 | None,
            "fields_filled_from_qr": ["irn", "buyer_gstin"],
            "fields_mismatched": [
                {"field": "invoice_no", "ocr_value": "...", "qr_value": "..."}
            ],
        }
    """
    if not qr_report:
        return {
            "qr_status": "not_run",
            "qr_reason": None,
            "qr_confidence": None,
            "fields_filled_from_qr": [],
            "fields_mismatched": [],
        }

    return {
        "qr_status": qr_report.get("qr_status"),
        "qr_reason": qr_report.get("qr_reason"),
        "qr_confidence": qr_report.get("qr_confidence"),
        "fields_filled_from_qr": qr_report.get("fields_filled_from_qr") or [],
        "fields_mismatched": qr_report.get("fields_mismatched") or [],
    }


def _build_metadata_section(
    invoice: Optional[dict],
    metadata: Optional[dict] = None,
) -> dict:
    """
    Build the "metadata" section: pipeline-level info (model name, OCR
    engine, page count, ...) merged with the per-field confidence scores
    already produced by field_resolver.resolve_fields() / qr_validator.

    The well-known keys (model_used, ocr_engine, page_count) always appear
    first for schema stability. Any additional keys the caller passes in
    `metadata` (e.g. validation_errors, is_valid) are merged in as-is
    rather than silently dropped, so callers can extend metadata without
    needing changes here.
    """
    invoice = invoice or {}
    metadata = metadata or {}

    known_keys = {"model_used", "ocr_engine", "page_count"}

    section = {
        "model_used": metadata.get("model_used"),
        "ocr_engine": metadata.get("ocr_engine"),
        "page_count": metadata.get("page_count"),
        "confidence": invoice.get("confidence") or {},
    }

    extras = {k: v for k, v in metadata.items() if k not in known_keys}
    section.update(extras)

    return section


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_invoice_result(
    image_path: str,
    invoice: dict,
    tables: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
    processing_time: Optional[float] = None,
    qr_report: Optional[dict] = None,
) -> dict:
    """
    Assemble the final structured result dict for a single invoice.

    This is the single source of truth for the export schema -- everything
    `export_invoice_json` writes to disk comes from this function, so there
    is exactly one place that knows the JSON layout.

    Parameters
    ----------
    image_path : str
        Path to the source invoice image (used to derive invoice_name).
    invoice : dict
        Output of field_resolver.resolve_fields(), enriched by
        qr_validator.validate_and_merge() / ensure_no_nulls().
    tables : list[dict], optional
        Output of table_extractor.extract_tables().
    metadata : dict, optional
        Extra pipeline metadata (model_used, ocr_engine, page_count, ...).
    processing_time : float, optional
        Total pipeline runtime in seconds.
    qr_report : dict, optional
        Output of qr_validator.validate_and_merge() -- the report half,
        not the mutated invoice.

    Returns
    -------
    dict
        Fully structured, JSON-serialisable invoice result.
    """
    invoice_name = Path(image_path).stem

    return {
        "invoice_name": invoice_name,
        "source_image": str(image_path),
        "processing_time": processing_time,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "header": _build_header_section(invoice),
        "tables": _build_tables_section(tables),
        "qr_validation": _build_qr_validation_section(qr_report),
        "metadata": _build_metadata_section(invoice, metadata),
    }


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------

def _resolve_output_path(invoice_name: str, output_dir: str) -> Path:
    """
    Build a non-clobbering output path for `invoice_name` inside
    `output_dir`, creating the directory if it does not exist.

    If `{invoice_name}.json` is already taken, a numeric suffix is
    appended (`{invoice_name}_2.json`, `{invoice_name}_3.json`, ...) so
    previously written results are never overwritten -- this is what lets
    multiple invoices (or re-runs of the same invoice) coexist.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = out_dir / f"{invoice_name}.json"
    if not candidate.exists():
        return candidate

    counter = 2
    while True:
        candidate = out_dir / f"{invoice_name}_{counter}.json"
        if not candidate.exists():
            return candidate
        counter += 1


def export_invoice_json(
    image_path: str,
    invoice: dict,
    tables: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
    processing_time: Optional[float] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    qr_report: Optional[dict] = None,
) -> Path:
    """
    Assemble and write the final invoice extraction result to a pretty
    printed JSON file.

    Parameters
    ----------
    image_path : str
        Path to the source invoice image. Used to derive the output
        filename (e.g. "invoice_001.png" -> "invoice_001.json").
    invoice : dict
        Output of field_resolver.resolve_fields(), enriched by qr_validator.
    tables : list[dict], optional
        Output of table_extractor.extract_tables().
    metadata : dict, optional
        Extra pipeline metadata to include (model_used, ocr_engine, ...).
    processing_time : float, optional
        Total pipeline runtime in seconds.
    output_dir : str, optional
        Directory the JSON file is written into. Defaults to "outputs_new".
        Created automatically via pathlib if it does not exist.
    qr_report : dict, optional
        Output of qr_validator.validate_and_merge() (the report dict).

    Returns
    -------
    pathlib.Path
        Path to the written JSON file.
    """
    result = build_invoice_result(
        image_path=image_path,
        invoice=invoice,
        tables=tables,
        metadata=metadata,
        processing_time=processing_time,
        qr_report=qr_report,
    )

    output_path = _resolve_output_path(result["invoice_name"], output_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False, default=str)

    return output_path


# ---------------------------------------------------------------------------
# Quick smoke test (python json_exporter.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy_invoice = {
        "supplier_gstin": "29ABCDE1234F1Z5",
        "buyer_gstin": "27XYZPQ9876K1Z3",
        "invoice_no": "INV-2025-001",
        "invoice_date": "2025-09-30",
        "irn": ("a1b2c3d4e5f6" * 5) + "aabb",
        "gstins_found": ["29ABCDE1234F1Z5", "27XYZPQ9876K1Z3"],
        "confidence": {
            "supplier_gstin": 0.95,
            "buyer_gstin": 0.90,
            "invoice_no": 0.85,
            "invoice_date": 0.80,
            "irn": 0.99,
        },
    }

    dummy_tables = [
        {
            "table_number": 1,
            "crop_path": "table_crops/dummy_table_1.png",
            "dataframe": pd.DataFrame(
                [
                    {"Description": "Room Charge", "Qty": "1", "Amount": "5000"},
                    {"Description": "GST", "Qty": "", "Amount": "900"},
                ]
            ),
        }
    ]

    dummy_qr_report = {
        "qr_status": "success",
        "qr_reason": None,
        "qr_confidence": 0.93,
        "fields_filled_from_qr": ["irn"],
        "fields_mismatched": [],
    }

    out_path = export_invoice_json(
        image_path="dummy_invoice.png",
        invoice=dummy_invoice,
        tables=dummy_tables,
        metadata={"model_used": "Newend.pt", "ocr_engine": "tesseract+easyocr"},
        processing_time=2.34,
        output_dir="outputs_test",
        qr_report=dummy_qr_report,
    )

    print(f"Wrote: {out_path}")
    print(out_path.read_text(encoding="utf-8"))