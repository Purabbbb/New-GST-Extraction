import re
import sys
import time

from header_extractor import extract_header
from table_extractor import extract_tables
from field_resolver import resolve_fields
from tax_resolver import resolve_tax_details
from json_exporter import export_invoice_json
from qr_validator import decode_and_validate, validate_and_merge, ensure_no_nulls

# ---------------------------------------------------------------------------
# Config – override via CLI arg or edit here
# ---------------------------------------------------------------------------

DEFAULT_IMAGE = "1_page_invoice/AAHIN260643061_page_1.png"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

_pipeline_start = time.time()

# ---------------------------------------------------------------------------
# STEP 1 – Header extraction
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 1 : HEADER EXTRACTION")
print("=" * 80)

header = extract_header(IMAGE_PATH)

print(f"Header crop       : {header['header_crop_path']}")
print(f"First table starts: y={header['first_table_y']} px")
print(f"Last table ends   : y={header['last_table_y']} px")

print("\n" + "-" * 40)
print("RAW OCR TEXT")
print("-" * 40)
print(header["ocr_text"])

# ---------------------------------------------------------------------------
# STEP 2 – Field Resolution
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 2 : FIELD RESOLUTION")
print("=" * 80)

invoice = resolve_fields(header["ocr_text"])

CORE_FIELDS = [
    "supplier_gstin",
    "buyer_gstin",
    "vendor_name",
    "buyer_name",
    "invoice_no",
    "invoice_date",
    "currency",
    "irn",
]

for field in CORE_FIELDS:
    value = invoice.get(field)
    print(f"{field:<20}: {value if value else '(not found)'}")

print("\nGSTINs Found:")
print(invoice.get("gstins_found", []))

print("\n" + "-" * 40)
print("CONFIDENCE SCORES")
print("-" * 40)

confidence = invoice.get("confidence", {})

for field, score in confidence.items():

    bar = "#" * int(score * 20)

    print(
        f"{field:<20}: {score:.2f} [{bar:<20}]"
    )

# ---------------------------------------------------------------------------
# STEP 2.5 – QR Decode + Cross-Validation
# ---------------------------------------------------------------------------
# Decodes the QR on the invoice (GST e-invoice QR / UPI QR / plain URL) and
# uses it two ways:
#   1. Fills any OCR field that came back None/empty.
#   2. Flags (without overwriting) any field where OCR and QR disagree.

print("\n" + "=" * 80)
print("STEP 2.5 : QR DECODE + CROSS-VALIDATION")
print("=" * 80)

qr_result = decode_and_validate(IMAGE_PATH)

print(f"QR status : {qr_result.get('status')}")

if qr_result.get("status") == "success":
    print(f"QR confidence : {qr_result.get('confidence')}")
    print(f"QR raw fields : {qr_result.get('data')}")
else:
    print(f"QR reason : {qr_result.get('reason')}")
    print(f"QR detail : {qr_result.get('detail')}")

invoice, qr_report = validate_and_merge(invoice, qr_result)

if qr_report["fields_filled_from_qr"]:
    print("\nFields filled from QR (OCR had no value):")
    for field in qr_report["fields_filled_from_qr"]:
        print(f"  - {field:<20}: {invoice.get(field)}")
else:
    print("\nNo OCR fields needed filling from QR.")

if qr_report["fields_mismatched"]:
    print("\nWARNING - OCR / QR mismatches detected:")
    for mismatch in qr_report["fields_mismatched"]:
        print(
            f"  - {mismatch['field']:<20}: "
            f"OCR='{mismatch['ocr_value']}'  vs  QR='{mismatch['qr_value']}'"
        )
else:
    print("\nNo mismatches between OCR and QR values.")

# ---------------------------------------------------------------------------
# GSTIN-specific override: for supplier_gstin / buyer_gstin, prefer QR over
# OCR whenever they disagree. QR is decoded from structured e-invoice data
# and carries no ambiguity; OCR's GSTIN resolution is a line-proximity
# heuristic that can misattribute values under two-column "Supplier |
# Recipient" layouts (the exact failure mode that produced the swap here).
# We still sanity-check the QR value looks like a real GSTIN before trusting
# it, so a QR decode error can't silently corrupt a correct OCR read.
# ---------------------------------------------------------------------------

_GSTIN_SHAPE_RE = re.compile(r'^[0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$')

_GSTIN_OVERRIDE_FIELDS = {"supplier_gstin", "buyer_gstin"}

gstin_overrides = []

for mismatch in qr_report["fields_mismatched"]:
    field = mismatch["field"]
    if field not in _GSTIN_OVERRIDE_FIELDS:
        continue
    qr_val = mismatch["qr_value"]
    if qr_val and _GSTIN_SHAPE_RE.match(qr_val.strip().upper()):
        invoice[field] = qr_val.strip().upper()
        invoice.setdefault("confidence", {})[field] = max(
            invoice.get("confidence", {}).get(field, 0.0), qr_result.get("confidence", 0.95)
        )
        gstin_overrides.append((field, mismatch["ocr_value"], qr_val))

if gstin_overrides:
    print("\nGSTIN fields overridden with QR value (QR outranks OCR for GSTIN):")
    for field, ocr_val, qr_val in gstin_overrides:
        print(f"  - {field:<20}: OCR had '{ocr_val}', now using QR '{qr_val}'")

# ---------------------------------------------------------------------------
# Determine whether GST is intra-state or inter-state
# ---------------------------------------------------------------------------

supplier = invoice.get("supplier_gstin")
buyer = invoice.get("buyer_gstin")

same_state = False

if supplier and buyer and len(supplier) >= 2 and len(buyer) >= 2:

    same_state = supplier[:2] == buyer[:2]

# ---------------------------------------------------------------------------
# STEP 3 – Table Extraction
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 3 : TABLE EXTRACTION")
print("=" * 80)

tables = extract_tables(IMAGE_PATH)

if not tables:

    print("No tables found.")

else:

    for table in tables:

        df = table["dataframe"]

        print(f"\nTable {table['table_number']}")
        print(f"Crop    : {table['crop_path']}")
        print(f"Rows    : {len(df)}")
        print(f"Columns : {list(df.columns)}")

        print("\nPreview\n")

        print(df.head(10).to_string(index=False))

        print("-" * 80)

# ---------------------------------------------------------------------------
# STEP 4 – Tax Resolution
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 4 : TAX RESOLUTION")
print("=" * 80)

tax_details = resolve_tax_details(
    tables,
    same_state=same_state
)

for key, value in tax_details.items():

    print(f"{key:<15}: {value}")

# ---------------------------------------------------------------------------
# Merge tax details into invoice
# Do NOT overwrite existing values with None.
#
# NOTE: this now LOGS any overwrite of an already-present value instead of
# silently replacing it. Previously a tax_resolver bug that double-counted
# subtotal/total_amount across tables silently clobbered a CORRECT
# QR-filled total_amount with no trace in the output -- that's how the 2x
# totals bug slipped through undetected. tax_resolver's table-classified
# ("grand_total"-sourced) figures are still the most authoritative single
# source for these fields, so we keep taking them, but now you'll see it
# in the console if a future regression disagrees with QR/OCR.
# ---------------------------------------------------------------------------

_tax_source = tax_details.get("table_used", "unknown")

for key, value in tax_details.items():

    if key == "table_used" or value is None:
        continue

    existing = invoice.get(key)

    if existing not in (None, "", 0) and existing != value:
        print(
            f"  NOTE: {key} changed from '{existing}' to '{value}' "
            f"(tax_resolver source: {_tax_source})"
        )

    invoice[key] = value

# ---------------------------------------------------------------------------
# Merge confidence (optional)
# ---------------------------------------------------------------------------

invoice.setdefault("confidence", {})

for field, value in tax_details.items():

    if value is not None:

        invoice["confidence"].setdefault(field, 1.0)

# ---------------------------------------------------------------------------
# Final null-safety pass
# ---------------------------------------------------------------------------
# After OCR + QR + tax resolution, anything still missing gets an explicit
# placeholder rather than a raw null, so the exported JSON never has to be
# special-cased downstream for "field missing" vs "field null".

print("\n" + "=" * 80)
print("STEP 4.5 : NULL-SAFETY PASS")
print("=" * 80)

before_fill = {
    field: invoice.get(field) for field in
    ["supplier_gstin", "buyer_gstin", "vendor_name", "buyer_name",
     "invoice_no", "invoice_date", "currency", "irn",
     "subtotal", "cgst", "sgst", "igst", "total_amount"]
}

invoice = ensure_no_nulls(invoice)

filled_by_safety_net = [
    field for field, old_value in before_fill.items()
    if (old_value is None or (isinstance(old_value, str) and old_value.strip() == ""))
]

if filled_by_safety_net:
    print("Fields with no OCR or QR value, set to placeholder:")
    for field in filled_by_safety_net:
        print(f"  - {field:<20}: {invoice.get(field)}")
else:
    print("All core fields resolved by OCR/QR/tax steps -- no placeholders needed.")

# ---------------------------------------------------------------------------
# STEP 5 – JSON Export
# ---------------------------------------------------------------------------

elapsed = round(time.time() - _pipeline_start, 3)

json_path = export_invoice_json(

    image_path=IMAGE_PATH,

    invoice=invoice,

    tables=tables,

    metadata={
        "model_used": "Newend.pt",
        "ocr_engine": "tesseract+easyocr",
    },

    processing_time=elapsed,

    qr_report=qr_report,
)

print("\n" + "=" * 80)
print("STEP 5 : JSON EXPORT")
print("=" * 80)

print(f"Saved JSON : {json_path}")

print("\nPipeline completed successfully.")