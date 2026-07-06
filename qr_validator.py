"""
qr_validator.py
================
Bridges qr_invoice_decoder.decode_invoice_qr() with the OCR-based invoice
dict produced by field_resolver.resolve_fields().

Responsibilities
----------------
1. Decode the QR code on the invoice image.
2. Normalise whatever keys the QR uses (GST e-invoice QR, UPI QR, plain
   URL/JSON QR, ...) onto our internal field names.
3. Fill any OCR field that came back None/empty with the QR value.
4. Flag (but never silently overwrite) any field where OCR and QR
   *both* have a value and they disagree.
5. Provide a final `ensure_no_nulls()` safety net so the fields the
   pipeline promises are never null in the exported JSON, even if
   neither OCR nor QR could find them.

This module does NOT re-implement QR decoding -- it only consumes the
dict returned by qr_invoice_decoder.decode_invoice_qr().
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from qr_invoice_decoder import decode_invoice_qr

# ---------------------------------------------------------------------------
# Field <-> QR key mapping
# ---------------------------------------------------------------------------
# Keys are matched after normalisation (lowercase, strip non-alphanumerics),
# so "Seller Gstin", "SellerGstin", "seller_gstin" all match the same alias.

FIELD_QR_ALIASES: Dict[str, List[str]] = {
    "supplier_gstin": ["sellergstin", "gstin1", "sellerid", "vendorgstin"],
    "buyer_gstin": ["buyergstin", "gstin2", "buyerid", "recipientgstin"],
    "vendor_name": ["sellername", "sellerlglnm", "sellertrdnm", "vendorname"],
    "buyer_name": ["buyername", "buyerlglnm", "buyertrdnm", "recipientname"],
    "invoice_no": ["docno", "invno", "invoiceno", "billno"],
    "invoice_date": ["docdt", "invdt", "invoicedate", "billdate"],
    "irn": ["irn"],
    "total_amount": ["totinvval", "totalvalue", "invval", "totamt", "amount"],
}

# Fields where exact string equality is too strict (dates/amounts might be
# formatted differently by OCR vs QR) -- these get a looser comparison.
_LOOSE_COMPARE_FIELDS = {"invoice_date", "total_amount"}

CORE_NULL_SAFE_FIELDS: List[str] = [
    "supplier_gstin",
    "buyer_gstin",
    "vendor_name",
    "buyer_name",
    "invoice_no",
    "invoice_date",
    "currency",
    "irn",
    "subtotal",
    "cgst",
    "sgst",
    "igst",
    "total_amount",
]

DEFAULT_PLACEHOLDER = "Not Available"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _normalize_value(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


def _build_qr_lookup(qr_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten + normalise QR field keys for alias matching."""
    lookup = {}
    for k, v in (qr_fields or {}).items():
        if isinstance(v, (dict, list)):
            continue
        lookup[_normalize_key(k)] = v
    return lookup


def _get_qr_value(qr_lookup: Dict[str, Any], aliases: List[str]) -> Optional[str]:
    for alias in aliases:
        if alias in qr_lookup:
            val = qr_lookup[alias]
            if val is not None and str(val).strip() != "":
                return str(val).strip()
    return None


def _values_match(field: str, ocr_value: str, qr_value: str) -> bool:
    if field in _LOOSE_COMPARE_FIELDS:
        # Compare digits only -- forgiving of "01/09/2025" vs "2025-09-01"
        # and "5,000.00" vs "5000.0"
        ocr_digits = re.sub(r"[^0-9]", "", ocr_value)
        qr_digits = re.sub(r"[^0-9]", "", qr_value)
        return ocr_digits == qr_digits
    return _normalize_value(ocr_value) == _normalize_value(qr_value)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def decode_and_validate(image_path: str) -> Dict[str, Any]:
    """
    Runs the QR decoder once. Wrapper so callers (and tests) don't need to
    import qr_invoice_decoder directly.
    """
    return decode_invoice_qr(image_path)


def validate_and_merge(
    invoice: Dict[str, Any],
    qr_result: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Cross-validates `invoice` (OCR output) against `qr_result` (output of
    decode_invoice_qr) and fills OCR nulls from the QR where possible.

    Returns
    -------
    (invoice, report)
        invoice : the same dict, mutated in place, also returned for
                  convenience.
        report  : dict describing what happened, e.g.

            {
                "qr_status": "success" | "failed",
                "qr_reason": None | "no_qr_detected" | ...,
                "qr_confidence": 0.91 | None,
                "fields_filled_from_qr": ["irn", "buyer_gstin"],
                "fields_mismatched": [
                    {"field": "invoice_no", "ocr_value": "...", "qr_value": "..."}
                ],
            }
    """
    invoice.setdefault("confidence", {})

    report: Dict[str, Any] = {
        "qr_status": qr_result.get("status"),
        "qr_reason": qr_result.get("reason"),
        "qr_confidence": qr_result.get("confidence"),
        "fields_filled_from_qr": [],
        "fields_mismatched": [],
    }

    if qr_result.get("status") != "success":
        return invoice, report

    qr_fields = qr_result.get("data") or {}
    qr_lookup = _build_qr_lookup(qr_fields)

    for field, aliases in FIELD_QR_ALIASES.items():
        qr_value = _get_qr_value(qr_lookup, aliases)

        if qr_value is None:
            continue

        ocr_value = invoice.get(field)
        ocr_value = str(ocr_value).strip() if ocr_value not in (None, "") else None

        if ocr_value is None:
            invoice[field] = qr_value
            invoice["confidence"][field] = max(
                invoice["confidence"].get(field, 0.0), 0.9
            )
            report["fields_filled_from_qr"].append(field)
        elif not _values_match(field, ocr_value, qr_value):
            report["fields_mismatched"].append(
                {
                    "field": field,
                    "ocr_value": ocr_value,
                    "qr_value": qr_value,
                }
            )

    return invoice, report


def ensure_no_nulls(
    invoice: Dict[str, Any],
    core_fields: Optional[List[str]] = None,
    placeholder: str = DEFAULT_PLACEHOLDER,
) -> Dict[str, Any]:
    """
    Final safety net: guarantees that none of `core_fields` are None/empty
    in the returned invoice dict. Anything still missing after OCR + QR
    is set to `placeholder` and given confidence 0.0, so downstream JSON
    consumers never have to special-case null vs "genuinely not found".
    """
    fields = core_fields if core_fields is not None else CORE_NULL_SAFE_FIELDS
    invoice.setdefault("confidence", {})

    for field in fields:
        value = invoice.get(field)
        is_missing = value is None or (isinstance(value, str) and value.strip() == "")
        if is_missing:
            invoice[field] = placeholder
            invoice["confidence"][field] = 0.0

    return invoice
