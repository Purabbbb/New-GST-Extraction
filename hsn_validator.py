"""
hsn_validator.py

Validates HSN/SAC codes appearing on an extracted invoice against a master
reference list (an Excel workbook of valid HSN/SAC codes + descriptions).

STATUS: scaffolded and tested against your invoice's own extracted tables,
but NOT yet wired to a real master file -- no .xlsx was actually uploaded
alongside table_extractor.py / tax_resolver.py / field_resolver.py in this
conversation. Once you upload the master-code workbook, tell me the sheet
name (if more than one) and I'll confirm the auto-detected column mapping
matches it -- the auto-detection below is a sensible default, not a
guarantee, until it's run against your real file.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

import pandas as pd

_CODE_COL_KEYWORDS = ["hsn", "sac", "code"]
_DESC_COL_KEYWORDS = ["description", "desc", "particulars", "service", "commodity"]

# HSN/SAC codes are 4, 6, or 8 digit numeric strings
_HSN_CODE_RE = re.compile(r"^\d{4,8}$")


def _detect_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in keywords):
            return col
    return None


def load_hsn_master(xlsx_path: str, sheet_name=0) -> Dict[str, str]:
    """
    Loads a master HSN/SAC code list from an Excel file.
    Returns {code: description} for every row with a plausible code.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str)

    code_col = _detect_column(df, _CODE_COL_KEYWORDS)
    desc_col = _detect_column(df, _DESC_COL_KEYWORDS)

    if code_col is None:
        raise ValueError(
            f"Could not find an HSN/SAC code column in {xlsx_path}. "
            f"Columns found: {list(df.columns)}"
        )

    master: Dict[str, str] = {}

    for _, row in df.iterrows():
        raw_code = re.sub(r"\D", "", str(row[code_col]).strip())
        if not raw_code or not _HSN_CODE_RE.match(raw_code):
            continue
        desc = str(row[desc_col]).strip() if desc_col else ""
        master[raw_code] = desc

    return master


def extract_codes_from_tables(tables: List[Dict]) -> Set[str]:
    """
    Pulls every HSN/SAC-shaped numeric token out of the extracted invoice
    tables. Handles two layouts seen in this pipeline's own output:

      1. A dedicated "HSN/SAC CODE" column (table 2 in this invoice).
      2. The code embedded inside a description cell, e.g.
         "Room Supplement 996311" / "996311 Accommodation"
         (table 1's "SACI" column in this invoice).
    """
    codes: Set[str] = set()

    for table in tables:
        df = table["dataframe"]
        if df.empty:
            continue

        hsn_cols = [
            c for c in df.columns
            if "hsn" in str(c).lower() or "sac" in str(c).lower()
        ]

        for col in hsn_cols:
            for value in df[col].dropna():
                for m in re.finditer(r"\b(\d{4,8})\b", str(value)):
                    codes.add(m.group(1))

        if not hsn_cols:
            for col in df.columns:
                name = str(col).lower()
                if any(k in name for k in ("description", "item", "particulars", "saci")):
                    for value in df[col].dropna():
                        for m in re.finditer(r"\b(\d{6})\b", str(value)):
                            codes.add(m.group(1))

    return codes


def validate_hsn_codes(tables: List[Dict], master: Dict[str, str]) -> Dict:
    """
    Cross-checks every HSN/SAC code found on the invoice against the master
    list.
    """
    found = extract_codes_from_tables(tables)

    valid = sorted(c for c in found if c in master)
    invalid = sorted(c for c in found if c not in master)

    return {
        "codes_found": sorted(found),
        "valid_codes": valid,
        "invalid_codes": invalid,
        "descriptions": {c: master[c] for c in valid},
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python hsn_validator.py <hsn_master.xlsx> [invoice_result.json]")
        sys.exit(1)

    master = load_hsn_master(sys.argv[1])
    print(f"Loaded {len(master)} HSN/SAC codes from master list.")

    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            data = json.load(f)
        tables = [
            {"dataframe": pd.DataFrame(t["rows"], columns=t["headers"])}
            for t in data.get("tables", [])
        ]
        result = validate_hsn_codes(tables, master)
        print(json.dumps(result, indent=2))