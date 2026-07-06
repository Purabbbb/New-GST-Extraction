import re
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 15-character Indian GSTIN pattern
_GSTIN_REGEX = re.compile(
    r'^[0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$'
)

# IRN: exactly 64 hexadecimal characters
_IRN_REGEX = re.compile(r'^[0-9a-fA-F]{64}$')

# Date formats the resolver may produce.
# The resolver normalises to YYYY-MM-DD (ISO-8601), but we accept the two
# legacy formats as well so the validator remains robust against partial
# migrations or manual overrides.
_DATE_FORMATS = [
    "%Y-%m-%d",   # ISO-8601 – primary output of resolve_fields
    "%d/%m/%Y",   # dd/mm/YYYY
    "%d-%m-%Y",   # dd-mm-YYYY
]

# Minimum number of columns a table must have to be considered valid
_MIN_TABLE_COLUMNS = 2


# ---------------------------------------------------------------------------
# GSTIN validation
# ---------------------------------------------------------------------------

def validate_gstin(gstin: Optional[str]) -> bool:
    """
    Return True if `gstin` matches the Indian GSTIN format.

    Parameters
    ----------
    gstin : str | None
        The GSTIN string to validate.  None always returns False.
    """
    if not gstin:
        return False
    return bool(_GSTIN_REGEX.match(gstin.strip().upper()))


# ---------------------------------------------------------------------------
# IRN validation
# ---------------------------------------------------------------------------

def validate_irn(irn: Optional[str]) -> bool:
    """
    Return True if `irn` is a valid 64-character hexadecimal string.

    Parameters
    ----------
    irn : str | None
        The IRN string to validate.  None always returns False.
    """
    if not irn:
        return False
    return bool(_IRN_REGEX.match(irn.replace(" ", "").strip()))


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------

def validate_date(date_str: Optional[str]) -> bool:
    """
    Return True if `date_str` can be parsed as a valid calendar date.

    Accepted formats: YYYY-MM-DD (primary), dd/mm/YYYY, dd-mm-YYYY.

    Parameters
    ----------
    date_str : str | None
        The date string to validate.  None always returns False.
    """
    if not date_str:
        return False

    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(date_str.strip(), fmt)
            return True
        except ValueError:
            continue

    return False


# ---------------------------------------------------------------------------
# Field-level validation
# ---------------------------------------------------------------------------

def validate_fields(invoice: dict) -> list[str]:
    """
    Validate all header fields extracted by resolve_fields().

    Parameters
    ----------
    invoice : dict
        The dictionary returned by resolve_fields().

    Returns
    -------
    list[str]
        Human-readable error messages.  Empty list means all fields are valid.
    """
    errors: list[str] = []

    # --- Supplier GSTIN ---
    supplier = invoice.get("supplier_gstin")
    if not supplier:
        errors.append("Supplier GSTIN missing")
    elif not validate_gstin(supplier):
        errors.append(f"Invalid supplier GSTIN: {supplier}")

    # --- Buyer GSTIN ---
    buyer = invoice.get("buyer_gstin")
    if not buyer:
        errors.append("Buyer GSTIN missing")
    elif not validate_gstin(buyer):
        errors.append(f"Invalid buyer GSTIN: {buyer}")

    # --- Supplier and buyer must differ ---
    if supplier and buyer and supplier.upper() == buyer.upper():
        errors.append(
            f"Supplier and buyer GSTIN are identical: {supplier}"
        )

    # --- Invoice number ---
    invoice_no = invoice.get("invoice_no")
    if not invoice_no:
        errors.append("Invoice number missing")

    # --- Invoice date ---
    invoice_date = invoice.get("invoice_date")
    if not invoice_date:
        errors.append("Invoice date missing")
    elif not validate_date(invoice_date):
        errors.append(f"Invalid invoice date: {invoice_date!r}")

    # --- IRN (optional – only validated when present) ---
    irn = invoice.get("irn")
    if irn and not validate_irn(irn):
        errors.append(f"Invalid IRN (must be 64 hex chars): {irn[:16]}...")

    # --- Low-confidence warnings (not hard errors) ---
    confidence = invoice.get("confidence", {})
    for field_name, score in confidence.items():
        if score > 0 and score < 0.60:
            logger.warning(
                "Low confidence for %s: %.2f – manual review recommended",
                field_name, score,
            )

    return errors


# ---------------------------------------------------------------------------
# Table validation
# ---------------------------------------------------------------------------

def validate_tables(tables: list[dict]) -> list[str]:
    """
    Validate the list of tables returned by table_extractor.extract_tables().

    Parameters
    ----------
    tables : list[dict]
        Each dict must have keys: table_number, dataframe.

    Returns
    -------
    list[str]
        Human-readable error messages.  Empty list means tables look valid.
    """
    errors: list[str] = []

    if not tables:
        errors.append("No tables extracted from invoice")
        return errors

    for table in tables:
        table_num = table.get("table_number", "?")
        df = table.get("dataframe")

        if df is None or df.empty:
            errors.append(f"Table {table_num} is empty")
            continue

        if len(df.columns) < _MIN_TABLE_COLUMNS:
            errors.append(
                f"Table {table_num} has too few columns "
                f"({len(df.columns)} found, {_MIN_TABLE_COLUMNS} required)"
            )

    return errors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_invoice(invoice: dict, tables: list[dict]) -> list[str]:
    """
    Run all validations and return a combined list of errors.

    Parameters
    ----------
    invoice : dict
        Structured invoice fields from resolve_fields().
    tables  : list[dict]
        Table list from extract_tables().

    Returns
    -------
    list[str]
        All validation errors across fields and tables.
        An empty list means the invoice passed all checks.
    """
    errors: list[str] = []
    errors.extend(validate_fields(invoice))
    errors.extend(validate_tables(tables))
    return errors