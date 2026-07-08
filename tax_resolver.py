"""
tax_resolver.py

Resolves tax information from extracted invoice tables.

Input:
    tables = extract_tables(...)

Output:
    {
        "subtotal": ...,
        "cgst": ...,
        "sgst": ...,
        "igst": ...,
        "vat": ...,
        "cess": ...,
        "total_tax": ...,
        "total_amount": ...,
        "table_used": ...,   # NEW - which table role subtotal/total came from
    }

------------------------------------------------------------------------------
FIX (see explanation): the old version summed subtotal/total_amount across
ALL tables. Invoices like this one emit the SAME taxable value / grand total
at multiple levels of granularity (line-items table, HSN rollup table, final
summary table). Blindly summing across all of them double-, sometimes
triple-, counts. This version classifies each table by its column signature
and only trusts ONE table per figure, in priority order:

    grand_total  (has "Taxable Value" + "Total Inv. Value"-style columns)
        -> used for subtotal + total_amount + CGST/SGST/IGST/CESS if present
    hsn_summary  (has HSN/SAC + Sales columns)
        -> used only as a fallback for subtotal/tax if no grand_total table
    line_items   (per-row item detail)
        -> used only as a last-resort fallback (old row-summing behaviour)
------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def safe_float(value) -> Optional[float]:
    """
    Converts OCR strings to float.

    Examples
    --------
    "1,234.50" -> 1234.5
    "₹1,234.50" -> 1234.5
    """

    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    value = value.replace(",", "")

    value = re.sub(r"[^\d.-]", "", value)

    if value == "":
        return None

    try:
        return float(value)
    except Exception:
        return None


def extract_amount_rate(text):
    """
    Extracts amount and rate from OCR text.

    Examples
    --------
    "601.02 18.0"
        -> (601.02,18)

    "5.0 120.0"
        -> (120.0,5)

    "244.38 5.0"
        -> (244.38,5)
    """

    if text is None:
        return None, None

    numbers = re.findall(r"\d+\.\d+|\d+", str(text))

    if len(numbers) < 2:
        return None, None

    a = float(numbers[0])
    b = float(numbers[1])

    # Smaller value is usually the rate
    if a <= 40 and b > 40:
        rate = a
        amount = b
    else:
        amount = a
        rate = b

    return amount, rate


def find_column(df, keywords):

    for col in df.columns:

        name = str(col).lower()

        for key in keywords:

            if key in name:
                return col

    return None


def find_column_all(df, *, must_include: list[str]):
    """Like find_column, but every keyword in `must_include` must appear
    in the column name (used to disambiguate e.g. 'State CESS' vs
    'Central CESS', which both contain 'cess')."""

    for col in df.columns:

        name = str(col).lower()

        if all(key in name for key in must_include):
            return col

    return None


# ----------------------------------------------------------
# Table classification
# ----------------------------------------------------------

def classify_table(df: pd.DataFrame) -> str:
    """
    Classifies a table by its column signature so the caller can avoid
    summing the same figures twice across tables that represent the same
    data at different levels of granularity.

    Returns one of: "grand_total", "hsn_summary", "line_items"
    """

    cols = " | ".join(str(c).lower() for c in df.columns)

    if "taxable value" in cols and (
        "total inv" in cols or "grand total" in cols
    ):
        return "grand_total"

    if ("hsn" in cols or "sac" in cols) and "sales" in cols:
        return "hsn_summary"

    return "line_items"


def _sum_column(df: pd.DataFrame, col) -> Optional[float]:

    if col is None:
        return None

    total = 0.0
    found = False

    for _, row in df.iterrows():
        value = safe_float(row[col])
        if value is not None:
            total += value
            found = True

    return round(total, 2) if found else None


def _extract_direct_tax_fields(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Reads CGST/SGST/IGST/State CESS/Central CESS as plain numeric
    columns (single value per cell), summing rows within this ONE table.
    Used for grand_total / hsn_summary tables where these are direct
    figures rather than 'rate amount' pairs."""

    cgst_col = find_column(df, ["cgst"])
    sgst_col = find_column(df, ["sgst", "ugst"])
    igst_col = find_column(df, ["igst"])
    state_cess_col = find_column_all(df, must_include=["state", "cess"])
    central_cess_col = find_column_all(df, must_include=["central", "cess"])
    vat_col = find_column(df, ["vat"])

    state_cess = _sum_column(df, state_cess_col)
    central_cess = _sum_column(df, central_cess_col)

    cess = None
    if state_cess is not None or central_cess is not None:
        cess = round((state_cess or 0.0) + (central_cess or 0.0), 2)

    return {
        "cgst": _sum_column(df, cgst_col),
        "sgst": _sum_column(df, sgst_col),
        "igst": _sum_column(df, igst_col),
        "cess": cess,
        "vat": _sum_column(df, vat_col),
    }


# ----------------------------------------------------------
# Fallback: old row-by-row line-items logic
# (only used when no grand_total / hsn_summary table exists)
# ----------------------------------------------------------

def _resolve_from_line_items(
    tables: List[Dict],
    same_state: bool,
) -> Dict:

    subtotal = 0.0
    gst_amount = 0.0
    vat_amount = 0.0
    cess_amount = 0.0
    total_amount = 0.0

    for table in tables:

        df: pd.DataFrame = table["dataframe"]

        if df.empty or classify_table(df) != "line_items":
            continue

        gst_col = find_column(df, ["gst"])
        vat_col = find_column(df, ["vat"])
        cess_col = find_column(df, ["cess"])
        value_col = find_column(df, ["value", "amount", "taxable", "item"])
        total_col = find_column(df, ["total", "grand"])

        for _, row in df.iterrows():

            if value_col:
                value = safe_float(row[value_col])
                if value:
                    subtotal += value

            if gst_col:
                amount, _rate = extract_amount_rate(row[gst_col])
                if amount:
                    gst_amount += amount

            if vat_col:
                amount, _rate = extract_amount_rate(row[vat_col])
                if amount:
                    vat_amount += amount

            if cess_col:
                amount, _rate = extract_amount_rate(row[cess_col])
                if amount:
                    cess_amount += amount

            if total_col:
                amount = safe_float(row[total_col])
                if amount:
                    total_amount += amount

    cgst = sgst = igst = None

    if gst_amount > 0:
        if same_state:
            cgst = round(gst_amount / 2, 2)
            sgst = round(gst_amount / 2, 2)
            igst = 0.0
        else:
            cgst = 0.0
            sgst = 0.0
            igst = round(gst_amount, 2)

    return {
        "subtotal": round(subtotal, 2) if subtotal else None,
        "cgst": cgst,
        "sgst": sgst,
        "igst": igst,
        "vat": round(vat_amount, 2) if vat_amount else None,
        "cess": round(cess_amount, 2) if cess_amount else None,
        "total_tax": (
            round(gst_amount + vat_amount + cess_amount, 2)
            if (gst_amount + vat_amount + cess_amount)
            else None
        ),
        "total_amount": round(total_amount, 2) if total_amount else None,
        "table_used": "line_items (fallback)",
    }


# ----------------------------------------------------------
# Main Resolver
# ----------------------------------------------------------

def resolve_tax_details(
    tables: List[Dict],
    same_state: bool = True,
) -> Dict:

    grand_total_tables = []
    hsn_summary_tables = []

    for table in tables:
        df = table["dataframe"]
        if df.empty:
            continue
        role = classify_table(df)
        if role == "grand_total":
            grand_total_tables.append(df)
        elif role == "hsn_summary":
            hsn_summary_tables.append(df)

    # ------------------------------------------------------
    # Priority 1: a single "grand total" summary table
    # ------------------------------------------------------
    if grand_total_tables:

        df = grand_total_tables[0]  # first one wins if >1 (rare/ambiguous)

        value_col = find_column(df, ["taxable"])
        total_col = find_column(df, ["total inv", "grand total", "total"])

        subtotal = _sum_column(df, value_col)
        total_amount = _sum_column(df, total_col)

        tax_fields = _extract_direct_tax_fields(df)

        total_tax = None
        parts = [tax_fields["cgst"], tax_fields["sgst"], tax_fields["igst"], tax_fields["cess"]]
        if any(p is not None for p in parts):
            total_tax = round(sum(p or 0.0 for p in parts), 2)

        return {
            "subtotal": subtotal,
            "cgst": tax_fields["cgst"],
            "sgst": tax_fields["sgst"],
            "igst": tax_fields["igst"],
            "vat": tax_fields["vat"],
            "cess": tax_fields["cess"],
            "total_tax": total_tax,
            "total_amount": total_amount,
            "table_used": "grand_total",
        }

    # ------------------------------------------------------
    # Priority 2: an HSN-level summary table (no final total available)
    # ------------------------------------------------------
    if hsn_summary_tables:

        df = hsn_summary_tables[0]

        value_col = find_column(df, ["sales"])
        subtotal = _sum_column(df, value_col)

        tax_fields = _extract_direct_tax_fields(df)

        total_tax = None
        parts = [tax_fields["cgst"], tax_fields["sgst"], tax_fields["igst"], tax_fields["cess"]]
        if any(p is not None for p in parts):
            total_tax = round(sum(p or 0.0 for p in parts), 2)

        total_amount = None
        if subtotal is not None and total_tax is not None:
            total_amount = round(subtotal + total_tax, 2)

        return {
            "subtotal": subtotal,
            "cgst": tax_fields["cgst"],
            "sgst": tax_fields["sgst"],
            "igst": tax_fields["igst"],
            "vat": tax_fields["vat"],
            "cess": tax_fields["cess"],
            "total_tax": total_tax,
            "total_amount": total_amount,
            "table_used": "hsn_summary",
        }

    # ------------------------------------------------------
    # Priority 3: fall back to summing line-item rows (old behaviour,
    # but now scoped ONLY to tables classified as line_items so it can
    # never double-count against a summary table)
    # ------------------------------------------------------
    return _resolve_from_line_items(tables, same_state)


# ----------------------------------------------------------
# TEST
# ----------------------------------------------------------

if __name__ == "__main__":

    from table_extractor import extract_tables

    IMAGE = "invoice_test2.png"

    tables = extract_tables(IMAGE)

    tax = resolve_tax_details(
        tables,
        same_state=True,
    )

    print("\nResolved Tax Details\n")

    for k, v in tax.items():

        print(f"{k:15} : {v}")