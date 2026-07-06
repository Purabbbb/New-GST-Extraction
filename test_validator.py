"""
test_validator.py
=================
End-to-end integration test for the invoice validation pipeline.

Runs the full pipeline on a single image, reports any validation
errors found by validator.py, and exports the final structured
result to a JSON file via json_exporter.py.

Usage
-----
    python test_validator.py
    python test_validator.py path/to/invoice.png   # override image path
"""

import sys
import time
from header_extractor import extract_header
from table_extractor import extract_tables
from field_resolver import resolve_fields
from validator import validate_invoice
from json_exporter import export_invoice_json

# ---------------------------------------------------------------------------
# Config – override via CLI arg or edit here
# ---------------------------------------------------------------------------

DEFAULT_IMAGE = "dataset_new/AAHIN260642681_page_1.png"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

_pipeline_start = time.time()

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# 1. Extract header region and run OCR
header = extract_header(IMAGE_PATH)

# 2. Extract tables
tables = extract_tables(IMAGE_PATH)

# 3. Resolve structured fields from OCR text
#    resolve_fields() expects a plain string, not the header dict.
invoice = resolve_fields(header["ocr_text"])

# 4. Validate
errors = validate_invoice(invoice, tables)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION RESULT")
print("=" * 60)

if errors:
    print(f"\nFLAGGED  ({len(errors)} issue(s) found)\n")
    for err in errors:
        print(f"  - {err}")
else:
    print("\nVALID  – no issues detected.")

# Always show what was extracted so failures are easy to diagnose
print("\n" + "-" * 60)
print("EXTRACTED FIELDS")
print("-" * 60)
for key in ("supplier_gstin", "buyer_gstin", "invoice_no",
            "invoice_date", "irn", "total_amount"):
    print(f"  {key:<20} : {invoice.get(key)}")

# ---------------------------------------------------------------------------
# JSON export
#   validation_errors is passed through metadata so it rides along with
#   the rest of the extraction result in the same JSON file.
# ---------------------------------------------------------------------------

elapsed = round(time.time() - _pipeline_start, 3)

json_path = export_invoice_json(
    image_path=IMAGE_PATH,
    invoice=invoice,
    tables=tables,
    metadata={
        "model_used": "Newend.pt",
        "ocr_engine": "tesseract+easyocr",
        "validation_errors": errors,
        "is_valid": not errors,
    },
    processing_time=elapsed,
)

print("\n" + "-" * 60)
print(f"JSON RESULT SAVED : {json_path}")
print("-" * 60)