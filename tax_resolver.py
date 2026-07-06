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
        "total_amount": ...
    }
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


# ----------------------------------------------------------
# Main Resolver
# ----------------------------------------------------------

def resolve_tax_details(
    tables: List[Dict],
    same_state: bool = True,
) -> Dict:

    subtotal = 0.0

    gst_amount = 0.0

    vat_amount = 0.0

    cess_amount = 0.0

    total_amount = 0.0

    for table in tables:

        df: pd.DataFrame = table["dataframe"]

        if df.empty:
            continue

        gst_col = find_column(df, ["gst"])

        vat_col = find_column(df, ["vat"])

        cess_col = find_column(df, ["cess"])

        value_col = find_column(
            df,
            [
                "value",
                "amount",
                "taxable",
                "item",
            ],
        )

        total_col = find_column(
            df,
            [
                "total",
                "grand",
            ],
        )

        for _, row in df.iterrows():

            # -------------------------------
            # taxable value
            # -------------------------------

            if value_col:

                value = safe_float(row[value_col])

                if value:
                    subtotal += value

            # -------------------------------
            # GST
            # -------------------------------

            if gst_col:

                amount, rate = extract_amount_rate(
                    row[gst_col]
                )

                if amount:
                    gst_amount += amount

            # -------------------------------
            # VAT
            # -------------------------------

            if vat_col:

                amount, rate = extract_amount_rate(
                    row[vat_col]
                )

                if amount:
                    vat_amount += amount

            # -------------------------------
            # CESS
            # -------------------------------

            if cess_col:

                amount, rate = extract_amount_rate(
                    row[cess_col]
                )

                if amount:
                    cess_amount += amount

            # -------------------------------
            # Final Amount
            # -------------------------------

            if total_col:

                amount = safe_float(row[total_col])

                if amount:
                    total_amount += amount

    # ------------------------------------------------------
    # Split GST into CGST/SGST or IGST
    # ------------------------------------------------------

    cgst = None
    sgst = None
    igst = None

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

        "subtotal":
            round(subtotal, 2)
            if subtotal
            else None,

        "cgst":
            cgst,

        "sgst":
            sgst,

        "igst":
            igst,

        "vat":
            round(vat_amount, 2)
            if vat_amount
            else None,

        "cess":
            round(cess_amount, 2)
            if cess_amount
            else None,

        "total_tax":
            round(
                gst_amount +
                vat_amount +
                cess_amount,
                2,
            )
            if (
                gst_amount +
                vat_amount +
                cess_amount
            )
            else None,

        "total_amount":
            round(total_amount, 2)
            if total_amount
            else None,
    }


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