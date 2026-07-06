'''
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – edit these to extend coverage without touching logic
# ---------------------------------------------------------------------------

# Regex: Indian GSTIN  (15-char alphanumeric, state-code prefix 01-37/99)
_GSTIN_RE = re.compile(
    r'\b([0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b'
)

# IRN – exactly 64 hex characters (may have spaces due to OCR line-wrap)
_IRN_RE = re.compile(r'\b([0-9a-fA-F]{64})\b')

# Regex: currency amounts  (handles Rs, commas, decimals)
_AMOUNT_RE = re.compile(
    r'(?:Rs\.?\s*)?'
    r'([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?'   # 1,23,456.00
    r'|[0-9]{4,}(?:\.[0-9]{1,2})?)'                      # 123456 / 123456.00
)

# Regex: supported date formats
_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b(\d{2})[/-](\d{2})[/-](\d{4})\b'), 'dmy4'),   # dd/mm/yyyy
    (re.compile(r'\b(\d{4})[/-](\d{2})[/-](\d{2})\b'), 'ymd4'),   # yyyy-mm-dd
    (re.compile(r'\b(\d{2})[/-](\d{2})[/-](\d{2})\b'),  'dmy2'),  # dd/mm/yy
    (re.compile(
        r'\b(\d{1,2})[- ]+'
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
        r'[- ]+(\d{2,4})\b', re.IGNORECASE
    ), 'dMonY'),  # 1 Jan 2025 / 01-Jan-25 / 26-Sep-2025
]

_MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5,  'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}

# Keywords that precede supplier/seller information
_SUPPLIER_KEYWORDS: list[str] = [
    'seller', 'supplier', 'vendor', 'bill from', 'billed from',
    'ship from', 'shipped from', 'sold by', 'from',
    'gst registration', 'gst registration no', 'gst reg no',
    'service provider', 'manufacturer', 'issuer',
    'property gstn', 'hotel gstin',
]

# Keywords that precede buyer/recipient information
_BUYER_KEYWORDS: list[str] = [
    'buyer', 'beneficiary', 'recipient', 'customer', 'client',
    'bill to', 'billed to', 'ship to', 'shipped to', 'consignee',
    'purchaser', 'guest company', 'guest details', 'company name',
    'deliver to', 'sold to', 'gstn number', 'gstin number',
    'company gstin', 'billing gstin',
]

# Keywords that label the invoice number field
_INVOICE_NO_KEYWORDS: list[str] = [
    'tax invoice no', 'tax invoice number',
    'invoice no', 'invoice number', 'invoice #',
    'document no', 'document number', 'doc no',
    'bill no', 'bill number',
    'inv no', 'inv #',
]

# Keywords to AVOID when looking for invoice number
# (prevents picking up reservation/ref/room numbers)
_INVOICE_NO_AVOID_KEYWORDS: list[str] = [
    'reservation', 'ref no', 'reference no', 'reference number',
    'room no', 'room number', 'booking', 'folio',
    'confirmation no', 'confirmation number',
    'registration no',
]

# Words that, if the entire candidate matches, signal a false positive
_INVOICE_NO_REJECT_WORDS: frozenset[str] = frozenset({
    'b2b', 'b2c', 'tax', 'gst', 'igst', 'cgst', 'sgst',
    'invoice', 'original', 'duplicate', 'triplicate',
    'receipt', 'debit', 'credit', 'note', 'revised',
    'cp', 'ep', 'ap', 'map',  # hotel plan codes
})

# Keywords that label the invoice date field
_INVOICE_DATE_KEYWORDS: list[str] = [
    'invoice date', 'invoice dt',
    'tax invoice date', 'document date', 'doc date',
    'bill date', 'billing date', 'date of invoice',
    'date of issue', 'issue date', 'dated',
]

# Date keywords to avoid (arrival/departure/check-in/check-out)
_DATE_AVOID_KEYWORDS: list[str] = [
    'arrival', 'departure', 'check in', 'check out',
    'check-in', 'check-out', 'checkin', 'checkout',
    'ack date', 'acknowledgement date', 'print date',
]

# Keywords that label the grand total field
_TOTAL_KEYWORDS: list[str] = [
    'grand total', 'invoice value', 'invoice total',
    'net amount', 'net payable', 'total amount',
    'total payable', 'total invoice value', 'amount payable',
    'total due', 'balance due', 'payable amount',
    'final amount', 'net invoice value', 'total inv. value',
    'total bill amount', 'gross payable',
]

# ---------------------------------------------------------------------------
# Per-field search windows
# ---------------------------------------------------------------------------
# OCR from some hotel/ERP systems places ALL labels in a block at the top,
# and ALL values in a corresponding block below.  The value block can start
# 10-20 lines after its label.  These windows are tuned conservatively but
# large enough to handle that layout.

_WINDOW_INVOICE_NO   = 20   # invoice numbers can be far below the label row
_WINDOW_INVOICE_DATE = 18   # dates can appear after several label lines
_WINDOW_GSTIN        = 15   # buyer GSTIN may be many lines below "Company Name"
_WINDOW_IRN          = 25   # IRN sometimes follows Ack No / Ack Date lines
_WINDOW_TOTAL        = 20   # grand total can appear after item-detail tables

# Kept as a default for find_value_after_keywords (backward-compat wrapper)
_CONTEXT_WINDOW = 10

# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------
# Set to True to print a detailed candidate trace for every extracted field.
# Useful during development; set False in production.

DEBUG_EXTRACTION: bool = False


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    """A single candidate value extracted during scanning."""
    value: str
    score: float          # 0.0 – 1.0; higher is better
    keyword: str          # which keyword triggered extraction
    offset: int           # line distance from keyword
    source_line: str      # the line it came from (for debug)
    method: str = ''


@dataclass
class _ExtractionResult:
    """Holds a raw extracted value with its confidence score."""
    value: Optional[str] = None
    confidence: float = 0.0
    method: str = ''          # for debugging / logging


@dataclass
class ResolvedFields:
    """Structured output from the resolver."""
    supplier_gstin: Optional[str] = None
    buyer_gstin: Optional[str] = None
    invoice_no: Optional[str] = None
    invoice_date: Optional[str] = None   # ISO-8601: YYYY-MM-DD
    irn: Optional[str] = None
    #total_amount: Optional[str] = None
    gstins_found: list[str] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'supplier_gstin': self.supplier_gstin,
            'buyer_gstin':    self.buyer_gstin,
            'invoice_no':     self.invoice_no,
            'invoice_date':   self.invoice_date,
            'irn':            self.irn,
            #'total_amount':   self.total_amount,
            'gstins_found':   self.gstins_found,
            'confidence':     self.confidence,
        }


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(raw_text: str) -> list[str]:
    """
    Normalise raw OCR text into a clean list of non-empty lines.

    Steps
    -----
    1. Collapse Windows-style line endings.
    2. Strip leading/trailing whitespace from every line.
    3. Normalise runs of whitespace inside a line to a single space.
    4. Drop blank lines.
    """
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')

    lines: list[str] = []
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        line = re.sub(r'[ \t]+', ' ', line)  # normalise internal spaces
        if line:
            lines.append(line)

    return lines


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def search_pattern(
    lines: list[str],
    pattern: re.Pattern,
    *,
    group: int = 1,
) -> list[str]:
    """
    Scan all lines for `pattern` and return every unique match (group `group`).
    """
    seen: set[str] = set()
    results: list[str] = []
    for line in lines:
        for m in pattern.finditer(line):
            val = m.group(group)
            if val not in seen:
                seen.add(val)
                results.append(val)
    return results


def _keyword_line_indices(lines: list[str], keywords: list[str]) -> list[int]:
    """
    Return the indices of all lines that contain any of the given keywords
    (case-insensitive, partial match).
    Sorted longest-keyword-first to prefer more specific matches.
    """
    sorted_kws = sorted(keywords, key=len, reverse=True)
    return [
        i for i, line in enumerate(lines)
        if any(kw in line.lower() for kw in sorted_kws)
    ]


def _inline_value_after_colon(line: str, keyword: str) -> Optional[str]:
    """
    For a line like "Invoice No : IBIS-73262 Document Date : 2025-10-11",
    extract the token(s) that appear immediately after `keyword` and its
    colon/separator, stopping at the next label pattern.

    Returns the extracted fragment or None if keyword not on this line.

    This handles multi-label-per-line OCR output correctly by isolating
    the value segment that belongs specifically to `keyword`.
    """
    line_lower = line.lower()
    kw_lower = keyword.lower()
    pos = line_lower.find(kw_lower)
    if pos == -1:
        return None

    # Advance past keyword
    after = line[pos + len(keyword):]

    # Strip leading separator characters (colon, dot, dash, space, pipe)
    after = re.sub(r'^[\s:.\-|#]+', '', after)

    if not after:
        return None

    # Stop at the next "Word(s) : " pattern (another label).
    # Requires at least one space before the separator so we don't split
    # inside a value like "IBIS-73262" (where '-' is part of the token).
    next_label = re.search(
        r'\b[A-Z][A-Za-z]{2,}(?:\s+[A-Za-z]{2,})*\s+[:\-]\s+[A-Z0-9]',
        after,
    )
    if next_label:
        after = after[:next_label.start()].strip()

    return after.strip() if after.strip() else None


# ---------------------------------------------------------------------------
# Candidate cleaning
# ---------------------------------------------------------------------------

# Pattern that matches common OCR noise at the START of a token:
#   + INV123   →  INV123
#   : INV123   →  INV123
#   > INV123   →  INV123
#   • INV123   →  INV123
#   1 18/10/25 →  18/10/25   (stray leading digit(s) before a date)
_OCR_LEADING_NOISE_RE = re.compile(
    r'^'
    r'(?:'
    r'[+>:;\-•·|#*@~]+\s*'   # punctuation / bullets / arrows
    r'|'
    r'\d{1,3}\s+'             # stray 1–3 digits followed by space
    r')*'
)


def clean_candidate(raw: str) -> str:
    """
    Strip common OCR noise from the start of a candidate value.

    Handles:
    - "+ FM0636BIL0004439"  →  "FM0636BIL0004439"
    - ": INV-12345"         →  "INV-12345"
    - "> B0025/37930"       →  "B0025/37930"
    - "1 18/10/25"          →  "18/10/25"   (stray leading digit before date)
    - "• INV123"            →  "INV123"

    Does NOT strip leading digits that are part of the value itself
    (e.g. "372502" stays "372502" because there is no trailing space separator).
    """
    cleaned = _OCR_LEADING_NOISE_RE.sub('', raw.strip())
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Core candidate-based extraction engine
# ---------------------------------------------------------------------------

def _extract_best_candidate(
    lines: list[str],
    keywords: list[str],
    value_pattern: re.Pattern,
    scorer: Callable[[str, str, int], float],
    *,
    window: int = _CONTEXT_WINDOW,
    group: int = 1,
    avoid_keywords: Optional[list[str]] = None,
    field_name: str = '',           # used only in debug output
) -> _ExtractionResult:
    """
    Candidate-based extraction: collect every possible value across all
    keyword hits and context windows, then return the highest-scoring one.

    Parameters
    ----------
    lines          : preprocessed OCR lines
    keywords       : ordered list of keyword strings to search for
    value_pattern  : regex to match candidate values
    scorer         : callable(value, source_line, offset) → float score 0–1
                     Higher means better candidate.
    window         : how many lines after a keyword hit to scan
    group          : capture group index in value_pattern
    avoid_keywords : lines whose ONLY content matches these are skipped in the
                     context window (e.g. "Arrival Date" when seeking invoice date)
    field_name     : label for debug output (e.g. 'Invoice Number')

    Returns
    -------
    _ExtractionResult with best value and confidence score.

    Notes on candidate cleaning
    ---------------------------
    Before scoring, every raw candidate is passed through clean_candidate()
    which strips OCR noise such as leading '+', ':', '>', stray single
    digits, bullets, etc.  This is what allows "+ FM0636BIL0004439" and
    "1 18/10/25" to be extracted correctly.
    """
    sorted_kws = sorted(keywords, key=len, reverse=True)
    avoid_kws = [a.lower() for a in (avoid_keywords or [])]

    all_candidates: list[_Candidate] = []
    keyword_hit_lines: list[int] = []   # for debug

    for i, line in enumerate(lines):
        line_lower = line.lower()

        matched_kw = next(
            (kw for kw in sorted_kws if kw in line_lower), None
        )
        if matched_kw is None:
            continue

        keyword_hit_lines.append(i)

        # Skip if this line contains an avoid-keyword BUT does not contain
        # the primary keyword we are looking for.  This prevents lines like
        # "Invoice Date : 30/09/25 Arrival Date : 28/09/25" from being
        # rejected when we're seeking 'invoice date', while still suppressing
        # pure arrival/departure-date lines.
        if avoid_kws:
            has_avoid = any(a in line_lower for a in avoid_kws)
            has_primary = any(kw in line_lower for kw in sorted_kws)
            if has_avoid and not has_primary:
                continue

        # --- Strategy A: extract the inline fragment right after keyword:colon ---
        inline_frag = _inline_value_after_colon(line, matched_kw)
        if inline_frag:
            cleaned_frag = clean_candidate(inline_frag)
            for source in (inline_frag, cleaned_frag):
                for m in value_pattern.finditer(source):
                    raw_val = m.group(group).strip()
                    val = clean_candidate(raw_val)
                    if val:
                        score = scorer(val, line, 0)
                        score = min(1.0, score + 0.10)  # inline bonus
                        all_candidates.append(_Candidate(
                            value=val,
                            score=score,
                            keyword=matched_kw,
                            offset=0,
                            source_line=line,
                            method=f'inline-after-colon "{matched_kw}"',
                        ))

        # --- Strategy B: scan full same line and context window ---
        scan_range = range(0, min(window + 1, len(lines) - i))
        for offset in scan_range:
            candidate_line = lines[i + offset]
            candidate_line_lower = candidate_line.lower()

            # Skip context-window lines that ARE exclusively an avoid-keyword
            # (e.g. "Arrival Date" line in the window of "Invoice Date")
            # But do NOT skip the keyword-hit line itself (offset==0).
            if offset > 0 and avoid_kws:
                if any(a in candidate_line_lower for a in avoid_kws):
                    continue

            # Run regex on both the raw line and the cleaned version so that
            # artefacts like "+ FM0636BIL0004439" are captured.
            cleaned_line = clean_candidate(candidate_line)
            for source_line_variant in _unique_variants(candidate_line, cleaned_line):
                for m in value_pattern.finditer(source_line_variant):
                    raw_val = m.group(group).strip()
                    val = clean_candidate(raw_val)
                    if not val:
                        continue

                    base_score = scorer(val, candidate_line, offset)
                    distance_penalty = offset * 0.04
                    score = max(0.0, base_score - distance_penalty)

                    all_candidates.append(_Candidate(
                        value=val,
                        score=score,
                        keyword=matched_kw,
                        offset=offset,
                        source_line=candidate_line,
                        method=f'window[+{offset}] after "{matched_kw}"',
                    ))

    # --- Debug output ---
    if DEBUG_EXTRACTION and field_name:
        _debug_print(field_name, keyword_hit_lines, lines, all_candidates, window)

    if not all_candidates:
        return _ExtractionResult()

    best = max(all_candidates, key=lambda c: c.score)

    if best.score <= 0.0:
        return _ExtractionResult()

    logger.debug(
        '_extract_best_candidate[%s]: value=%r score=%.2f method=%s',
        field_name, best.value, best.score, best.method,
    )

    return _ExtractionResult(
        value=best.value,
        confidence=min(1.0, best.score),
        method=best.method,
    )


def _unique_variants(*strings: str) -> list[str]:
    """Return unique non-empty strings, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for s in strings:
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _debug_print(
    field_name: str,
    keyword_lines: list[int],
    lines: list[str],
    candidates: list['_Candidate'],
    window: int,
) -> None:
    """Print a structured candidate trace to stdout when DEBUG_EXTRACTION=True."""
    sep = '-' * 56
    print(f'\n{"="*56}')
    print(f'DEBUG: {field_name}')
    print(f'{"="*56}')

    if not keyword_lines:
        print('  Keyword: NOT FOUND IN DOCUMENT')
    else:
        for kl in keyword_lines:
            end = min(kl + window, len(lines) - 1)
            print(f'  Keyword at line {kl}: {lines[kl]!r}')
            print(f'  Search window   : lines {kl}–{end}')

    print(f'\n  {sep}')
    if candidates:
        # Deduplicate by value, keep highest score
        seen: dict[str, _Candidate] = {}
        for c in candidates:
            if c.value not in seen or c.score > seen[c.value].score:
                seen[c.value] = c
        ranked = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        print(f'  Candidates ({len(ranked)} unique):')
        for c in ranked[:10]:   # show top 10
            print(f'    {c.value:<30}  score={c.score:.2f}  [{c.method}]')
    else:
        print('  Candidates: NONE')

    if candidates:
        best = max(candidates, key=lambda c: c.score)
        if best.score > 0.0:
            print(f'\n  Selected: {best.value!r}  (score={best.score:.2f})')
        else:
            print('\n  Selected: NONE (all scored 0)')
    print(f'  {sep}')



# ---------------------------------------------------------------------------
# Backward-compatible wrapper (used by resolve_irn / resolve_total_amount)
# ---------------------------------------------------------------------------

def find_value_after_keywords(
    lines: list[str],
    keywords: list[str],
    value_pattern: re.Pattern,
    *,
    window: int = _CONTEXT_WINDOW,
    group: int = 1,
) -> _ExtractionResult:
    """
    Legacy helper kept for backward compatibility.

    Uses the new candidate engine with a simple distance-based scorer
    so existing callers (resolve_irn, resolve_total_amount) continue to work
    without changes.
    """
    def _simple_scorer(val: str, line: str, offset: int) -> float:
        return max(0.50, 0.90 - offset * 0.04)

    return _extract_best_candidate(
        lines, keywords, value_pattern, _simple_scorer,
        window=window, group=group,
    )


# ---------------------------------------------------------------------------
# GSTIN helpers
# ---------------------------------------------------------------------------

def find_all_gstins(lines: list[str]) -> list[str]:
    """Return all unique GSTINs found anywhere in the document, in order."""
    return search_pattern(lines, _GSTIN_RE)


def _gstins_in_window(lines: list[str], start: int, window: int) -> list[str]:
    """Extract all GSTINs from lines[start : start + window]."""
    seen: set[str] = set()
    results: list[str] = []
    for line in lines[start: start + window]:
        for m in _GSTIN_RE.finditer(line):
            val = m.group(1)
            if val not in seen:
                seen.add(val)
                results.append(val)
    return results


def _resolve_gstin_by_context(
    lines: list[str],
    keywords: list[str],
    all_gstins: list[str],
    *,
    exclude: Optional[str] = None,
    window: int = _WINDOW_GSTIN,
) -> _ExtractionResult:
    """
    Find a GSTIN by locating a keyword block then scanning ahead.

    Improvement over v1: scans ALL keyword hits and ALL candidates,
    then picks the one with the lowest distance to its keyword.
    """
    sorted_kws = sorted(keywords, key=len, reverse=True)

    best: Optional[_Candidate] = None

    for i, line in enumerate(lines):
        if not any(kw in line.lower() for kw in sorted_kws):
            continue

        for offset in range(0, min(window, len(lines) - i)):
            candidate_line = lines[i + offset]
            for m in _GSTIN_RE.finditer(candidate_line):
                gstin = m.group(1)
                if gstin == exclude:
                    continue
                score = max(0.55, 0.92 - offset * 0.04)
                c = _Candidate(
                    value=gstin,
                    score=score,
                    keyword=line,
                    offset=offset,
                    source_line=candidate_line,
                    method=f'gstin-context[+{offset}] line {i}',
                )
                if best is None or c.score > best.score:
                    best = c

    if best:
        return _ExtractionResult(
            value=best.value,
            confidence=best.score,
            method=best.method,
        )

    return _ExtractionResult()


# ---------------------------------------------------------------------------
# Field-specific resolvers
# ---------------------------------------------------------------------------

def resolve_supplier_gstin(
    lines: list[str],
    all_gstins: list[str],
    buyer_gstin: Optional[str] = None,
) -> _ExtractionResult:
    """
    Identify the supplier (seller/vendor) GSTIN.

    Strategy
    --------
    1. Context-window scan using supplier keywords.
    2. If only two GSTINs exist and buyer is already known, the remaining
       one is the supplier.
    3. Returns None rather than guessing when context is ambiguous.
    """
    result = _resolve_gstin_by_context(
        lines, _SUPPLIER_KEYWORDS, all_gstins, exclude=buyer_gstin
    )
    if result.value:
        return result

    if buyer_gstin and len(all_gstins) == 2:
        remaining = [g for g in all_gstins if g != buyer_gstin]
        if remaining:
            return _ExtractionResult(
                value=remaining[0],
                confidence=0.60,
                method='deduction (two GSTINs, buyer known)',
            )

    logger.debug('resolve_supplier_gstin: no supplier GSTIN identified')
    return _ExtractionResult()


def resolve_buyer_gstin(
    lines: list[str],
    all_gstins: list[str],
    supplier_gstin: Optional[str] = None,
) -> _ExtractionResult:
    """
    Identify the buyer (recipient/beneficiary) GSTIN.

    Strategy mirrors resolve_supplier_gstin but uses buyer keywords.
    Uses a wider search window because buyer GSTIN often appears several
    lines below "Company Name" or "GSTN Number".
    """
    result = _resolve_gstin_by_context(
        lines, _BUYER_KEYWORDS, all_gstins, exclude=supplier_gstin
    )
    if result.value:
        return result

    if supplier_gstin and len(all_gstins) == 2:
        remaining = [g for g in all_gstins if g != supplier_gstin]
        if remaining:
            return _ExtractionResult(
                value=remaining[0],
                confidence=0.60,
                method='deduction (two GSTINs, supplier known)',
            )

    logger.debug('resolve_buyer_gstin: no buyer GSTIN identified')
    return _ExtractionResult()


# ---------------------------------------------------------------------------
# Invoice number scorer
# ---------------------------------------------------------------------------

# Patterns that indicate a value is NOT an invoice number
_ROOM_RE = re.compile(r'^\d{1,4}$')           # pure short integer: room / floor
_ALL_ALPHA_SHORT = re.compile(r'^[A-Za-z]{1,4}$')  # plan codes: CP, EP, B2B

def _score_invoice_number(val: str, source_line: str, offset: int) -> float:
    """
    Score a candidate invoice number.

    Rules
    -----
    - Must contain at least one digit                      → hard requirement
    - Must be >= 4 characters                              → hard requirement
    - Must not be a reject word (B2B, CP, Tax, …)          → hard requirement
    - Pure short digits (1–4 digits only) → room/floor nr  → reject
    - Longer mixed alphanumeric → good candidate
    - Contains slash or dash → typical invoice format bonus
    - Presence of OCR artefacts like leading '+' → strip & penalise
    """
    val = val.strip().lstrip('+').strip()

    # Hard requirements
    if not re.search(r'\d', val):
        return 0.0
    if len(val) < 4:
        return 0.0
    if val.lower() in _INVOICE_NO_REJECT_WORDS:
        return 0.0
    if _ROOM_RE.match(val):
        return 0.0
    if _ALL_ALPHA_SHORT.match(val):
        return 0.0

    # Penalise values that look like dates
    for pat, _ in _DATE_PATTERNS:
        if pat.fullmatch(val):
            return 0.0

    score = 0.70  # base

    # Bonus: contains a letter + digit mix (classic invoice number format)
    if re.search(r'[A-Za-z]', val) and re.search(r'\d', val):
        score += 0.15

    # Bonus: contains separator characters common in invoice numbers
    if re.search(r'[/\-_]', val):
        score += 0.05

    # Bonus: longer values are more likely to be real invoice numbers
    if len(val) >= 8:
        score += 0.05

    # Penalise very long values (might be an IRN or address fragment)
    if len(val) > 35:
        score -= 0.30

    return min(1.0, score)


def resolve_invoice_number(lines: list[str]) -> _ExtractionResult:
    """
    Extract the invoice number.

    Handles layouts:
    - "Invoice No: INV123"                    (inline)
    - "Invoice No\\nINV123"                   (next line)
    - "Invoice No : IBIS-73262 Document Date : 2025-10-11 Room Number : 318"
      (multi-field single line – inline-after-colon isolates the right token)
    - "Category : B2B    Invoice No : 372502  Document Date : 25/09/2025"
      (all on one line)

    OCR artefact cleaning
    ---------------------
    Strips leading '+' characters that Tesseract sometimes prepends.
    """
    # Broad capture: alphanumeric token (with optional leading OCR noise chars).
    # The lookbehind also accepts '+', ':', '>' which are common Tesseract artefacts.
    # Note: clean_candidate() is applied to every match BEFORE scoring, so
    # "FM0636BIL0004439" is correctly extracted from "+ FM0636BIL0004439".
    _INV_VAL_RE = re.compile(
        r'(?:^|(?<=[+>:;\s\-]))([A-Z0-9][A-Z0-9/\-_]{2,34})',
        re.IGNORECASE,
    )

    result = _extract_best_candidate(
        lines,
        _INVOICE_NO_KEYWORDS,
        _INV_VAL_RE,
        _score_invoice_number,
        window=_WINDOW_INVOICE_NO,
        avoid_keywords=_INVOICE_NO_AVOID_KEYWORDS,
        field_name='Invoice Number',
    )

    if result.value:
        # Clean OCR artefact: strip leading '+'
        result.value = result.value.lstrip('+').strip()
        return result

    return _ExtractionResult()


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_date_string(raw: str) -> Optional[str]:
    """
    Parse a raw date string into ISO-8601 (YYYY-MM-DD).
    Returns None if parsing fails.
    """
    raw = raw.strip()

    for pattern, fmt in _DATE_PATTERNS:
        m = pattern.search(raw)
        if not m:
            continue

        try:
            if fmt == 'dmy4':
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif fmt == 'ymd4':
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif fmt == 'dmy2':
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                y += 2000 if y < 50 else 1900
            elif fmt == 'dMonY':
                d = int(m.group(1))
                mo = _MONTH_MAP.get(m.group(2).lower()[:3], 0)
                y = int(m.group(3))
                if y < 100:
                    y += 2000 if y < 50 else 1900
            else:
                continue

            return datetime(y, mo, d).strftime('%Y-%m-%d')

        except (ValueError, KeyError):
            continue

    return None


def _score_date(val: str, source_line: str, offset: int) -> float:
    """
    Score a date candidate.

    A value scores well only if it is actually parseable as a valid date.
    Arrival / departure dates are rejected via avoid_keywords at the caller.
    """
    parsed = _parse_date_string(val)
    if not parsed:
        return 0.0

    # Prefer recent dates (2020–2030) as invoice dates
    try:
        dt = datetime.strptime(parsed, '%Y-%m-%d')
        if 2015 <= dt.year <= 2035:
            score = 0.85
        else:
            score = 0.50  # very old or far-future date is suspicious
    except ValueError:
        score = 0.70

    return score


def resolve_invoice_date(lines: list[str]) -> _ExtractionResult:
    """
    Extract the invoice date, normalised to YYYY-MM-DD.

    Handles:
    - "Invoice Date: 30/09/2025"
    - "Invoice Date\\n30/09/2025"
    - "Document Date : 2025-09-30"
    - "Bill Date: 01-Jan-25"
    - "Invoice No : 372502  Document Date : 25/09/2025" (multi-field line)

    Arrival / departure / check-in / check-out dates are excluded via
    avoid_keywords so we never return a hotel stay date as the invoice date.
    """
    _ANY_DATE_RE = re.compile(
        r'('
        r'\d{1,2}[/-]\d{2}[/-]\d{2,4}'
        r'|\d{4}[/-]\d{2}[/-]\d{2}'
        r'|\d{1,2}[- ]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[- ]+\d{2,4}'
        r')',
        re.IGNORECASE,
    )

    result = _extract_best_candidate(
        lines,
        _INVOICE_DATE_KEYWORDS,
        _ANY_DATE_RE,
        _score_date,
        window=_WINDOW_INVOICE_DATE,
        avoid_keywords=_DATE_AVOID_KEYWORDS,
        field_name='Invoice Date',
    )

    if result.value:
        parsed = _parse_date_string(result.value)
        if parsed:
            return _ExtractionResult(
                value=parsed,
                confidence=result.confidence,
                method=result.method,
            )

    logger.debug('resolve_invoice_date: no date extracted')
    return _ExtractionResult()


# ---------------------------------------------------------------------------
# IRN resolver (with space-collapse for OCR line-wrap)
# ---------------------------------------------------------------------------

def _collapse_hex_fragments(lines: list[str], start: int, window: int) -> Optional[str]:
    """
    OCR sometimes splits a 64-char hex string across multiple lines or
    inserts spaces.  This function:
    1. Collects text from start..(start+window) lines.
    2. Strips all whitespace to produce a single string.
    3. Searches for any 64-char hex substring.

    Returns the first 64-char hex found, or None.
    """
    fragment = ''.join(lines[start: start + window])
    # Remove spaces
    compact = re.sub(r'\s+', '', fragment)
    m = re.search(r'[0-9a-fA-F]{64}', compact)
    return m.group(0) if m else None


def resolve_irn(lines: list[str]) -> _ExtractionResult:
    """
    Extract the IRN (Invoice Reference Number) – a 64-character hex string.

    Improvements over v1
    --------------------
    - Collapses spaces within the context window to handle OCR line-wrap.
    - Scans all IRN-label positions and picks the nearest valid hex.
    - Falls back to first unlabelled 64-hex in the document.
    """
    # First try: direct match of solid 64-char hex anywhere
    all_irns = search_pattern(lines, _IRN_RE)

    irn_label_indices = _keyword_line_indices(lines, ['irn', 'invoice reference'])

    if not irn_label_indices:
        if all_irns:
            return _ExtractionResult(
                value=all_irns[0],
                confidence=0.65,
                method='unlabelled 64-char hex',
            )
        return _ExtractionResult()

    best: Optional[_ExtractionResult] = None

    for label_idx in irn_label_indices:
        scan_window = _WINDOW_IRN

        # Strategy 1: look for solid hex in window
        for offset in range(0, min(scan_window, len(lines) - label_idx)):
            m = _IRN_RE.search(lines[label_idx + offset])
            if m:
                conf = max(0.60, 0.95 - offset * 0.05)
                candidate = _ExtractionResult(
                    value=m.group(1),
                    confidence=conf,
                    method=f'IRN label context[+{offset}]',
                )
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
                break

        # Strategy 2: collapse whitespace and search for fragmented hex
        if best is None:
            collapsed = _collapse_hex_fragments(lines, label_idx, scan_window)
            if collapsed:
                best = _ExtractionResult(
                    value=collapsed,
                    confidence=0.75,
                    method='IRN collapsed-whitespace match',
                )

    if best:
        return best

    # Final fallback: first unlabelled 64-hex in document
    if all_irns:
        return _ExtractionResult(
            value=all_irns[0],
            confidence=0.65,
            method='fallback unlabelled',
        )

    return _ExtractionResult()


# ---------------------------------------------------------------------------
# Total amount scorer and resolver
# ---------------------------------------------------------------------------

def _score_amount(val: str, source_line: str, offset: int) -> float:
    """
    Score a currency amount candidate.
    Prefers larger amounts (more likely to be the grand total than a line item).
    """
    try:
        numeric = float(val.replace(',', ''))
    except ValueError:
        return 0.0

    if numeric <= 0:
        return 0.0

    # Reasonable invoice total range (not a GST rate, not a room number)
    if numeric < 10:
        return 0.10  # probably a GST rate percentage
    if numeric < 100:
        return 0.40

    score = 0.75

    # Larger amounts score higher (grand total > line item)
    if numeric >= 1000:
        score += 0.10
    if numeric >= 10000:
        score += 0.05

    return min(1.0, score)


def resolve_total_amount(lines: list[str]) -> _ExtractionResult:
    """
    Extract the grand total / invoice value.

    Handles:
    - "Grand Total: Rs.1,23,456.00"
    - "Invoice Value\\n1,23,456"
    - "Net Amount  1,23,456.00"
    - "Total Inv. Value   11,800.00" (EzyInvoice layout)
    """
    result = _extract_best_candidate(
        lines,
        _TOTAL_KEYWORDS,
        _AMOUNT_RE,
        _score_amount,
        window=_WINDOW_TOTAL,
        field_name='Total Amount',
    )

    if result.value:
        result.value = result.value.replace(',', '').strip()
        return result

    return _ExtractionResult()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def resolve_fields(raw_ocr_text: str) -> dict:
    """
    Main entry point.  Accepts raw OCR text and returns a fully structured
    extraction result dictionary.

    Parameters
    ----------
    raw_ocr_text : str
        The raw string from Tesseract / PaddleOCR.

    Returns
    -------
    dict with keys:
        supplier_gstin, buyer_gstin, invoice_no, invoice_date,
        irn, total_amount, gstins_found, confidence
    """
    lines = preprocess_text(raw_ocr_text)

    if not lines:
        logger.warning('resolve_fields: received empty text')
        return ResolvedFields().to_dict()

    # --- Step 1: collect all GSTINs globally ---
    all_gstins = find_all_gstins(lines)
    logger.debug('GSTINs found: %s', all_gstins)

    # --- Step 2: supplier GSTIN (resolve first to enable exclusion in buyer) ---
    supplier_result = resolve_supplier_gstin(lines, all_gstins)

    # --- Step 3: buyer GSTIN (exclude supplier to prevent collision) ---
    buyer_result = resolve_buyer_gstin(
        lines, all_gstins, supplier_gstin=supplier_result.value
    )

    # --- Retry supplier excluding buyer (handles ambiguous document order) ---
    if not supplier_result.value and buyer_result.value:
        supplier_result = resolve_supplier_gstin(
            lines, all_gstins, buyer_gstin=buyer_result.value
        )

    # --- Step 4: invoice number ---
    inv_no_result = resolve_invoice_number(lines)

    # --- Step 5: invoice date ---
    inv_date_result = resolve_invoice_date(lines)

    # --- Step 6: IRN ---
    irn_result = resolve_irn(lines)

    # --- Step 7: total amount ---
    total_result = resolve_total_amount(lines)

    # --- Assemble output ---
    output = ResolvedFields(
        supplier_gstin=supplier_result.value,
        buyer_gstin=buyer_result.value,
        invoice_no=inv_no_result.value,
        invoice_date=inv_date_result.value,
        irn=irn_result.value,
        #total_amount=total_result.value,
        gstins_found=all_gstins,
        confidence={
            'supplier_gstin': supplier_result.confidence,
            'buyer_gstin':    buyer_result.confidence,
            'invoice_no':     inv_no_result.confidence,
            'invoice_date':   inv_date_result.confidence,
            'irn':            irn_result.confidence,
            #'total_amount':   total_result.confidence,
        },
    )

    logger.info(
        'resolve_fields complete | invoice_no=%s date=%s supplier=%s buyer=%s',
        output.invoice_no,
        output.invoice_date,
        output.supplier_gstin,
        output.buyer_gstin,
    )

    return output.to_dict()


# ---------------------------------------------------------------------------
# Quick smoke-test  (python field_resolver.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json
    logging.basicConfig(level=logging.DEBUG)

    # ---- Test 1: standard layout ----
    SAMPLE_STANDARD = """
    Tax Invoice

    Invoice No : INV-2025-001
    Document Date : 2025-09-30

    Supplier
    ABC Exports Pvt Ltd
    GSTIN: 29ABCDE1234F1Z5

    Bill To
    XYZ Traders
    GSTIN: 27XYZPQ9876K1Z3

    IRN:
    Ack No: 1234567890
    Ack Date: 30/09/2025
    a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6aabb

    Grand Total: Rs.1,23,456.00
    """

    # ---- Test 2: multi-column layout (the hard case) ----
    SAMPLE_MULTICOLUMN = """
    Tax Invoice

    Category : B2B     Invoice No : IBIS-73262    Document Date : 2025-10-11    Room Number : 318
    Document Type : Tax Invoice    Confirmation No : 607917070

    Supplier
    GSTIN : 06AABCI2732H1ZW
    IBIS Gurgaon Golf Course Road IT

    Recipient
    Guest Name : Ms. Arpita Mukherjee
    GSTIN: 06AAEFE1778R1ZU

    IRN : d6f43f55918067e36569c11f7b88e6bfccd08fdb82e4deef65a6f7ef7f17050e

    Grand Total: 27615.42
    """

    # ---- Test 3: Ramada layout (buyer GSTIN after "GSTN Number" label) ----
    SAMPLE_RAMADA = """
    TAX INVOICE

    Guest Name : MR MEHTA CHETAN
    Company Name : YATRA FOR BUSINESS PRIVATE LIMITED
    GSTN Number : 07AAEFE1763C1ZU
    Company Address : 3rd Floor, Unit No. 1, Vasant Arcade

    Invoice Date : 30/09/25
    Tax Invoice No. : F2551BIL26007081

    Property GSTN# : 06AADCG1506B1ZE

    IRN NO: ed7b42a4723f93c66e24fd990a96c1c5a4a1bc27dd96c023ecc1d68624f78f45

    Net Amount: 11819.80
    """

    # ---- Test 4: hotel label-block layout (labels first, values below) ----
    # This was the failing case: all labels appear before line 10, all values
    # appear after line 10.  Both invoice_no and invoice_date were missed
    # because the context window was too small.
    SAMPLE_HOTEL_LABELBLOCK = """
Invoice Number
Invoice Date
Room No

Room Type
Reservation #
Number of Pax
Arrival Date
Departure Date
Plan

Billing Instruction
Tariff

+ FM0636BIL0004439

1 18/10/25
1417
:STD

: 119811
21
17/10/25
18/10/25
"""

    for label, sample in [
        ('STANDARD', SAMPLE_STANDARD),
        ('MULTI-COLUMN', SAMPLE_MULTICOLUMN),
        ('RAMADA', SAMPLE_RAMADA),
        ('HOTEL LABEL-BLOCK', SAMPLE_HOTEL_LABELBLOCK),
    ]:
        print(f'\n{"="*60}')
        print(f'TEST: {label}')
        print('='*60)
        result = resolve_fields(sample)
        for k, v in result.items():
            if k != 'confidence':
                print(f'  {k:<20}: {v}')
        print('  confidence:')
        for k, v in result['confidence'].items():
            print(f'    {k:<18}: {v:.2f}')'''
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – edit these to extend coverage without touching logic
# ---------------------------------------------------------------------------

# Regex: Indian GSTIN  (15-char alphanumeric, state-code prefix 01-37/99)
_GSTIN_RE = re.compile(
    r'\b([0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b'
)

# IRN – exactly 64 hexadecimal characters (may have spaces due to OCR line-wrap)
_IRN_RE = re.compile(r'\b([0-9a-fA-F]{64})\b')

# Regex: currency amounts  (handles Rs, commas, decimals)
_AMOUNT_RE = re.compile(
    r'(?:Rs\.?\s*)?'
    r'([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?'   # 1,23,456.00
    r'|[0-9]{4,}(?:\.[0-9]{1,2})?)'                      # 123456 / 123456.00
)

# Regex: any alphanumeric "code"-shaped token (invoice numbers, PO numbers, ...)
# Note: the lookbehind also accepts '+', ':', '>' which are common Tesseract
# artefacts. clean_candidate() strips these before scoring, so
# "+ FM0636BIL0004439" still resolves to "FM0636BIL0004439".
_ALNUM_CODE_RE = re.compile(
    r'(?:^|(?<=[+>:;\s\-]))([A-Z0-9][A-Z0-9/\-_]{2,34})',
    re.IGNORECASE,
)

# Regex: matches an entire (already-isolated) line/fragment as one candidate.
# Used for free-text fields (names, payment terms, bank name/branch) where
# the value isn't a fixed-shape token but the rest of the line/fragment.
_FULL_LINE_RE = re.compile(r'(.+)')

# Regex: Indian bank IFSC code (4 letters + '0' + 6 alphanumeric)
_IFSC_RE = re.compile(r'\b([A-Z]{4}0[A-Z0-9]{6})\b')

# Regex: bank account number (8–20 digit run)
_ACCOUNT_NO_RE = re.compile(r'\b(\d{8,20})\b')

# Regex: supported date formats
_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b(\d{2})[/-](\d{2})[/-](\d{4})\b'), 'dmy4'),   # dd/mm/yyyy
    (re.compile(r'\b(\d{4})[/-](\d{2})[/-](\d{2})\b'), 'ymd4'),   # yyyy-mm-dd
    (re.compile(r'\b(\d{2})[/-](\d{2})[/-](\d{2})\b'),  'dmy2'),  # dd/mm/yy
    (re.compile(
        r'\b(\d{1,2})[- ]+'
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
        r'[- ]+(\d{2,4})\b', re.IGNORECASE
    ), 'dMonY'),  # 1 Jan 2025 / 01-Jan-25 / 26-Sep-2025
]

_MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5,  'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}

# Keywords that precede supplier/seller information (used for both the
# supplier GSTIN and the vendor display name)
_SUPPLIER_KEYWORDS: list[str] = [
    'seller', 'supplier', 'vendor', 'bill from', 'billed from',
    'ship from', 'shipped from', 'sold by', 'from',
    'gst registration', 'gst registration no', 'gst reg no',
    'service provider', 'manufacturer', 'issuer',
    'property gstn', 'hotel gstin',
]

# Keywords that precede buyer/recipient information (used for both the
# buyer GSTIN and the buyer display name)
_BUYER_KEYWORDS: list[str] = [
    'buyer', 'beneficiary', 'recipient', 'customer', 'client',
    'bill to', 'billed to', 'ship to', 'shipped to', 'consignee',
    'purchaser', 'guest company', 'guest details', 'company name',
    'deliver to', 'sold to', 'gstn number', 'gstin number',
    'company gstin', 'billing gstin',
]

# Keywords that label the invoice number field
_INVOICE_NO_KEYWORDS: list[str] = [
    'tax invoice no', 'tax invoice number',
    'invoice no', 'invoice number', 'invoice #',
    'document no', 'document number', 'doc no',
    'bill no', 'bill number',
    'inv no', 'inv #',
]

# Keywords to AVOID when looking for invoice number
# (prevents picking up reservation/ref/room numbers)
_INVOICE_NO_AVOID_KEYWORDS: list[str] = [
    'reservation', 'ref no', 'reference no', 'reference number',
    'room no', 'room number', 'booking', 'folio',
    'confirmation no', 'confirmation number',
    'registration no',
]

# Words that, if the entire candidate matches, signal a false positive
_INVOICE_NO_REJECT_WORDS: frozenset[str] = frozenset({
    'b2b', 'b2c', 'tax', 'gst', 'igst', 'cgst', 'sgst',
    'invoice', 'original', 'duplicate', 'triplicate',
    'receipt', 'debit', 'credit', 'note', 'revised',
    'cp', 'ep', 'ap', 'map',  # hotel plan codes
})

# Keywords that label the invoice date field
_INVOICE_DATE_KEYWORDS: list[str] = [
    'invoice date', 'invoice dt',
    'tax invoice date', 'document date', 'doc date',
    'bill date', 'billing date', 'date of invoice',
    'date of issue', 'issue date', 'dated',
]

# Date keywords to avoid (arrival/departure/check-in/check-out)
_DATE_AVOID_KEYWORDS: list[str] = [
    'arrival', 'departure', 'check in', 'check out',
    'check-in', 'check-out', 'checkin', 'checkout',
    'ack date', 'acknowledgement date', 'print date',
]

# Keywords that label the grand total field
_TOTAL_KEYWORDS: list[str] = [
    'grand total', 'invoice value', 'invoice total',
    'net amount', 'net payable', 'total amount',
    'total payable', 'total invoice value', 'amount payable',
    'total due', 'balance due', 'payable amount',
    'final amount', 'net invoice value', 'total inv. value',
    'total bill amount', 'gross payable',
]

# Keywords that label the pre-tax / taxable value
_SUBTOTAL_KEYWORDS: list[str] = [
    'subtotal', 'sub total', 'sub-total', 'taxable amount',
    'taxable value', 'total taxable value', 'net value',
    'basic amount', 'amount before tax', 'value before tax',
]

# Keywords for individual GST components
_CGST_KEYWORDS: list[str] = ['cgst', 'central gst', 'central tax']
_SGST_KEYWORDS: list[str] = ['sgst', 'state gst', 'state tax', 'ugst']
_IGST_KEYWORDS: list[str] = ['igst', 'integrated gst', 'integrated tax']

# Keywords that label a purchase-order reference
_PO_NUMBER_KEYWORDS: list[str] = [
    'po number', 'po no', 'p.o. no', 'p.o no', 'purchase order no',
    'purchase order number', 'order no', 'order number', 'po ref',
    'po reference',
]

# Keywords that label payment terms / due-date information
_PAYMENT_TERMS_KEYWORDS: list[str] = [
    'payment terms', 'payment term', 'terms of payment',
    'payment due', 'due date', 'credit period', 'terms & conditions',
    'terms and conditions',
]

# Bank-detail keywords
_BANK_ACCOUNT_KEYWORDS: list[str] = [
    'account no', 'account number', 'a/c no', 'a/c number',
    'bank account no', 'bank a/c no', 'account #',
]
_BANK_IFSC_KEYWORDS: list[str] = ['ifsc', 'ifsc code']
_BANK_NAME_KEYWORDS: list[str] = ['bank name', 'name of bank', 'bank :', 'bank']
_BANK_BRANCH_KEYWORDS: list[str] = ['branch name', 'branch address', 'branch']

# Currency detection: (regex, ISO-4217-ish code). Checked in order; a
# document can legitimately match more than one line, so we tally hits
# and pick the most frequent code rather than the first match.
_CURRENCY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'₹|\bRs\.?\b|\bINR\b', re.IGNORECASE), 'INR'),
    (re.compile(r'\$|\bUSD\b', re.IGNORECASE), 'USD'),
    (re.compile(r'€|\bEUR\b', re.IGNORECASE), 'EUR'),
    (re.compile(r'£|\bGBP\b', re.IGNORECASE), 'GBP'),
]

# Words that, on their own, are just a label rather than a company name.
# Used to reject vendor/buyer-name candidates that are really just the
# keyword line itself (e.g. "Supplier" with nothing else on the line).
_NAME_LABEL_REJECT_WORDS: frozenset[str] = frozenset(
    kw.lower() for kw in (_SUPPLIER_KEYWORDS + _BUYER_KEYWORDS)
) | {
    'address', 'details', 'particulars', 'invoice', 'tax invoice',
    'gstin', 'gst', 'guest', 'company',
}

# ---------------------------------------------------------------------------
# Per-field search windows
# ---------------------------------------------------------------------------
# OCR from some hotel/ERP systems places ALL labels in a block at the top,
# and ALL values in a corresponding block below.  The value block can start
# 10-20 lines after its label.  These windows are tuned conservatively but
# large enough to handle that layout.

_WINDOW_INVOICE_NO   = 20   # invoice numbers can be far below the label row
_WINDOW_INVOICE_DATE = 18   # dates can appear after several label lines
_WINDOW_GSTIN        = 15   # buyer GSTIN may be many lines below "Company Name"
_WINDOW_IRN          = 25   # IRN sometimes follows Ack No / Ack Date lines
_WINDOW_TOTAL        = 20   # grand total can appear after item-detail tables
_WINDOW_NAME         = 6    # company names usually sit right below their label
_WINDOW_PO           = 10   # PO number position varies by ERP template
_WINDOW_PAYMENT_TERMS = 6   # payment terms usually sit close to their label
_WINDOW_TAX          = 8    # tax breakdown lines are usually near each other
_WINDOW_BANK         = 8    # bank detail block is usually compact

# Kept as a default for find_value_after_keywords (backward-compat wrapper)
_CONTEXT_WINDOW = 10

# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------
# Set to True to print a detailed candidate trace for every extracted field.
# Useful during development; set False in production.

DEBUG_EXTRACTION: bool = False


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    """A single candidate value extracted during scanning."""
    value: str
    score: float          # 0.0 – 1.0; higher is better
    keyword: str          # which keyword triggered extraction
    offset: int           # line distance from keyword
    source_line: str      # the line it came from (for debug)
    method: str = ''


@dataclass
class _ExtractionResult:
    """Holds a raw extracted value with its confidence score."""
    value: Optional[str] = None
    confidence: float = 0.0
    method: str = ''          # for debugging / logging


@dataclass
class ResolvedFields:
    """Structured output from the resolver."""
    supplier_gstin: Optional[str] = None
    buyer_gstin: Optional[str] = None
    vendor_name: Optional[str] = None
    buyer_name: Optional[str] = None
    invoice_no: Optional[str] = None
    invoice_date: Optional[str] = None   # ISO-8601: YYYY-MM-DD
    total_amount: Optional[str] = None
    subtotal: Optional[str] = None
    cgst: Optional[str] = None
    sgst: Optional[str] = None
    igst: Optional[str] = None
    currency: Optional[str] = None
    po_number: Optional[str] = None
    payment_terms: Optional[str] = None
    bank_details: dict = field(default_factory=dict)
    irn: Optional[str] = None
    gstins_found: list[str] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'supplier_gstin': self.supplier_gstin,
            'buyer_gstin':    self.buyer_gstin,
            'vendor_name':    self.vendor_name,
            'buyer_name':     self.buyer_name,
            'invoice_no':     self.invoice_no,
            'invoice_date':   self.invoice_date,
            'total_amount':   self.total_amount,
            'subtotal':       self.subtotal,
            'cgst':           self.cgst,
            'sgst':           self.sgst,
            'igst':           self.igst,
            'currency':       self.currency,
            'po_number':      self.po_number,
            'payment_terms':  self.payment_terms,
            'bank_details':   self.bank_details,
            'irn':            self.irn,
            'gstins_found':   self.gstins_found,
            'confidence':     self.confidence,
        }


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(raw_text: str) -> list[str]:
    """
    Normalise raw OCR text into a clean list of non-empty lines.

    Steps
    -----
    1. Collapse Windows-style line endings.
    2. Strip leading/trailing whitespace from every line.
    3. Normalise runs of whitespace inside a line to a single space.
    4. Drop blank lines.
    """
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')

    lines: list[str] = []
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        line = re.sub(r'[ \t]+', ' ', line)  # normalise internal spaces
        if line:
            lines.append(line)

    return lines


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def search_pattern(
    lines: list[str],
    pattern: re.Pattern,
    *,
    group: int = 1,
) -> list[str]:
    """
    Scan all lines for `pattern` and return every unique match (group `group`).
    """
    seen: set[str] = set()
    results: list[str] = []
    for line in lines:
        for m in pattern.finditer(line):
            val = m.group(group)
            if val not in seen:
                seen.add(val)
                results.append(val)
    return results


def _keyword_line_indices(lines: list[str], keywords: list[str]) -> list[int]:
    """
    Return the indices of all lines that contain any of the given keywords
    (case-insensitive, partial match).
    Sorted longest-keyword-first to prefer more specific matches.
    """
    sorted_kws = sorted(keywords, key=len, reverse=True)
    return [
        i for i, line in enumerate(lines)
        if any(kw in line.lower() for kw in sorted_kws)
    ]


def _inline_value_after_colon(line: str, keyword: str) -> Optional[str]:
    """
    For a line like "Invoice No : IBIS-73262 Document Date : 2025-10-11",
    extract the token(s) that appear immediately after `keyword` and its
    colon/separator, stopping at the next label pattern.

    Returns the extracted fragment or None if keyword not on this line.

    This handles multi-label-per-line OCR output correctly by isolating
    the value segment that belongs specifically to `keyword`.
    """
    line_lower = line.lower()
    kw_lower = keyword.lower()
    pos = line_lower.find(kw_lower)
    if pos == -1:
        return None

    # Advance past keyword
    after = line[pos + len(keyword):]

    # Strip leading separator characters (colon, dot, dash, space, pipe)
    after = re.sub(r'^[\s:.\-|#]+', '', after)

    if not after:
        return None

    # Stop at the next "Word(s) : " pattern (another label).
    # Requires at least one space before the separator so we don't split
    # inside a value like "IBIS-73262" (where '-' is part of the token).
    next_label = re.search(
        r'\b[A-Z][A-Za-z]{2,}(?:\s+[A-Za-z]{2,})*\s+[:\-]\s+[A-Z0-9]',
        after,
    )
    if next_label:
        after = after[:next_label.start()].strip()

    return after.strip() if after.strip() else None


# ---------------------------------------------------------------------------
# Candidate cleaning
# ---------------------------------------------------------------------------

# Pattern that matches common OCR noise at the START of a token:
#   + INV123   →  INV123
#   : INV123   →  INV123
#   > INV123   →  INV123
#   • INV123   →  INV123
#   1 18/10/25 →  18/10/25   (stray leading digit(s) before a date)
_OCR_LEADING_NOISE_RE = re.compile(
    r'^'
    r'(?:'
    r'[+>:;\-•·|#*@~]+\s*'   # punctuation / bullets / arrows
    r'|'
    r'\d{1,3}\s+'             # stray 1–3 digits followed by space
    r')*'
)


def clean_candidate(raw: str) -> str:
    """
    Strip common OCR noise from the start of a candidate value.

    Handles:
    - "+ FM0636BIL0004439"  →  "FM0636BIL0004439"
    - ": INV-12345"         →  "INV-12345"
    - "> B0025/37930"       →  "B0025/37930"
    - "1 18/10/25"          →  "18/10/25"   (stray leading digit before date)
    - "• INV123"            →  "INV123"

    Does NOT strip leading digits that are part of the value itself
    (e.g. "372502" stays "372502" because there is no trailing space separator).
    """
    cleaned = _OCR_LEADING_NOISE_RE.sub('', raw.strip())
    return cleaned.strip()


# Matches a short "<Label> : " prefix at the start of a string, e.g.
# "Guest Name : Ms. Arpita Mukherjee" -> label="Guest Name", rest="Ms. ..."
_LABEL_PREFIX_RE = re.compile(r'^[A-Za-z][A-Za-z .]{1,30}?#?\s*:\s*(.+)$')


def _strip_label_prefix(text: str) -> str:
    """
    Strip a leading "<Label> : <value>" style prefix from a free-text
    candidate line.

    Free-text fields (vendor/buyer name, payment terms, bank name/branch)
    are captured as a WHOLE line via `_FULL_LINE_RE`. The line's anchor
    keyword (e.g. "Recipient") can differ from a label embedded further
    into the captured line (e.g. "Guest Name : Ms. Arpita Mukherjee"), so
    the raw candidate would otherwise include that embedded label. This
    isolates just the value portion.
    """
    text = text.strip()
    m = _LABEL_PREFIX_RE.match(text)
    return m.group(1).strip() if m else text


def _finalize_free_text_result(result: '_ExtractionResult') -> '_ExtractionResult':
    """
    Post-process a free-text extraction result: strip any embedded label
    prefix and reject values that turn out to have no letters left.
    """
    if not result.value:
        return result
    cleaned = _strip_label_prefix(result.value)
    if not cleaned or not re.search(r'[A-Za-z]', cleaned):
        return _ExtractionResult()
    result.value = cleaned
    return result


def _finalize_name_result(result: '_ExtractionResult') -> '_ExtractionResult':
    """
    Post-process a vendor/buyer name extraction result: strip any embedded
    label prefix, then reject values that turn out to actually be a GSTIN
    (e.g. "Property GSTN# : 06AADCG1506B1ZE" -> stripped to a bare GSTIN,
    which is clearly not a company name).
    """
    if not result.value:
        return result
    cleaned = _strip_label_prefix(result.value)
    if not cleaned or not re.search(r'[A-Za-z]', cleaned):
        return _ExtractionResult()
    if _GSTIN_RE.search(cleaned.upper()):
        return _ExtractionResult()
    result.value = cleaned
    return result


# ---------------------------------------------------------------------------
# Core candidate-based extraction engine
# ---------------------------------------------------------------------------

def _extract_best_candidate(
    lines: list[str],
    keywords: list[str],
    value_pattern: re.Pattern,
    scorer: Callable[[str, str, int], float],
    *,
    window: int = _CONTEXT_WINDOW,
    group: int = 1,
    avoid_keywords: Optional[list[str]] = None,
    field_name: str = '',           # used only in debug output
) -> _ExtractionResult:
    """
    Candidate-based extraction: collect every possible value across all
    keyword hits and context windows, then return the highest-scoring one.

    Parameters
    ----------
    lines          : preprocessed OCR lines
    keywords       : ordered list of keyword strings to search for
    value_pattern  : regex to match candidate values
    scorer         : callable(value, source_line, offset) → float score 0–1
                     Higher means better candidate.
    window         : how many lines after a keyword hit to scan
    group          : capture group index in value_pattern
    avoid_keywords : lines whose ONLY content matches these are skipped in the
                     context window (e.g. "Arrival Date" when seeking invoice date)
    field_name     : label for debug output (e.g. 'Invoice Number')

    Returns
    -------
    _ExtractionResult with best value and confidence score.

    Notes on candidate cleaning
    ---------------------------
    Before scoring, every raw candidate is passed through clean_candidate()
    which strips OCR noise such as leading '+', ':', '>', stray single
    digits, bullets, etc.  This is what allows "+ FM0636BIL0004439" and
    "1 18/10/25" to be extracted correctly.
    """
    sorted_kws = sorted(keywords, key=len, reverse=True)
    avoid_kws = [a.lower() for a in (avoid_keywords or [])]

    all_candidates: list[_Candidate] = []
    keyword_hit_lines: list[int] = []   # for debug

    for i, line in enumerate(lines):
        line_lower = line.lower()

        matched_kw = next(
            (kw for kw in sorted_kws if kw in line_lower), None
        )
        if matched_kw is None:
            continue

        keyword_hit_lines.append(i)

        # Skip if this line contains an avoid-keyword BUT does not contain
        # the primary keyword we are looking for.  This prevents lines like
        # "Invoice Date : 30/09/25 Arrival Date : 28/09/25" from being
        # rejected when we're seeking 'invoice date', while still suppressing
        # pure arrival/departure-date lines.
        if avoid_kws:
            has_avoid = any(a in line_lower for a in avoid_kws)
            has_primary = any(kw in line_lower for kw in sorted_kws)
            if has_avoid and not has_primary:
                continue

        # --- Strategy A: extract the inline fragment right after keyword:colon ---
        inline_frag = _inline_value_after_colon(line, matched_kw)
        if inline_frag:
            cleaned_frag = clean_candidate(inline_frag)
            for source in (inline_frag, cleaned_frag):
                for m in value_pattern.finditer(source):
                    raw_val = m.group(group).strip()
                    val = clean_candidate(raw_val)
                    if val:
                        score = scorer(val, line, 0)
                        score = min(1.0, score + 0.10)  # inline bonus
                        all_candidates.append(_Candidate(
                            value=val,
                            score=score,
                            keyword=matched_kw,
                            offset=0,
                            source_line=line,
                            method=f'inline-after-colon "{matched_kw}"',
                        ))

        # --- Strategy B: scan full same line and context window ---
        scan_range = range(0, min(window + 1, len(lines) - i))
        for offset in scan_range:
            candidate_line = lines[i + offset]
            candidate_line_lower = candidate_line.lower()

            # Skip context-window lines that ARE exclusively an avoid-keyword
            # (e.g. "Arrival Date" line in the window of "Invoice Date")
            # But do NOT skip the keyword-hit line itself (offset==0).
            if offset > 0 and avoid_kws:
                if any(a in candidate_line_lower for a in avoid_kws):
                    continue

            # Run regex on both the raw line and the cleaned version so that
            # artefacts like "+ FM0636BIL0004439" are captured.
            cleaned_line = clean_candidate(candidate_line)
            for source_line_variant in _unique_variants(candidate_line, cleaned_line):
                for m in value_pattern.finditer(source_line_variant):
                    raw_val = m.group(group).strip()
                    val = clean_candidate(raw_val)
                    if not val:
                        continue

                    base_score = scorer(val, candidate_line, offset)
                    distance_penalty = offset * 0.04
                    score = max(0.0, base_score - distance_penalty)

                    all_candidates.append(_Candidate(
                        value=val,
                        score=score,
                        keyword=matched_kw,
                        offset=offset,
                        source_line=candidate_line,
                        method=f'window[+{offset}] after "{matched_kw}"',
                    ))

    # --- Debug output ---
    if DEBUG_EXTRACTION and field_name:
        _debug_print(field_name, keyword_hit_lines, lines, all_candidates, window)

    if not all_candidates:
        return _ExtractionResult()

    best = max(all_candidates, key=lambda c: c.score)

    if best.score <= 0.0:
        return _ExtractionResult()

    logger.debug(
        '_extract_best_candidate[%s]: value=%r score=%.2f method=%s',
        field_name, best.value, best.score, best.method,
    )

    return _ExtractionResult(
        value=best.value,
        confidence=min(1.0, best.score),
        method=best.method,
    )


def _unique_variants(*strings: str) -> list[str]:
    """Return unique non-empty strings, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for s in strings:
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _debug_print(
    field_name: str,
    keyword_lines: list[int],
    lines: list[str],
    candidates: list['_Candidate'],
    window: int,
) -> None:
    """Print a structured candidate trace to stdout when DEBUG_EXTRACTION=True."""
    sep = '-' * 56
    print(f'\n{"="*56}')
    print(f'DEBUG: {field_name}')
    print(f'{"="*56}')

    if not keyword_lines:
        print('  Keyword: NOT FOUND IN DOCUMENT')
    else:
        for kl in keyword_lines:
            end = min(kl + window, len(lines) - 1)
            print(f'  Keyword at line {kl}: {lines[kl]!r}')
            print(f'  Search window   : lines {kl}–{end}')

    print(f'\n  {sep}')
    if candidates:
        # Deduplicate by value, keep highest score
        seen: dict[str, _Candidate] = {}
        for c in candidates:
            if c.value not in seen or c.score > seen[c.value].score:
                seen[c.value] = c
        ranked = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        print(f'  Candidates ({len(ranked)} unique):')
        for c in ranked[:10]:   # show top 10
            print(f'    {c.value:<30}  score={c.score:.2f}  [{c.method}]')
    else:
        print('  Candidates: NONE')

    if candidates:
        best = max(candidates, key=lambda c: c.score)
        if best.score > 0.0:
            print(f'\n  Selected: {best.value!r}  (score={best.score:.2f})')
        else:
            print('\n  Selected: NONE (all scored 0)')
    print(f'  {sep}')



# ---------------------------------------------------------------------------
# Backward-compatible wrapper (used by resolve_irn / resolve_total_amount)
# ---------------------------------------------------------------------------

def find_value_after_keywords(
    lines: list[str],
    keywords: list[str],
    value_pattern: re.Pattern,
    *,
    window: int = _CONTEXT_WINDOW,
    group: int = 1,
) -> _ExtractionResult:
    """
    Legacy helper kept for backward compatibility.

    Uses the new candidate engine with a simple distance-based scorer
    so existing callers (resolve_irn, resolve_total_amount) continue to work
    without changes.
    """
    def _simple_scorer(val: str, line: str, offset: int) -> float:
        return max(0.50, 0.90 - offset * 0.04)

    return _extract_best_candidate(
        lines, keywords, value_pattern, _simple_scorer,
        window=window, group=group,
    )


# ---------------------------------------------------------------------------
# GSTIN helpers
# ---------------------------------------------------------------------------

def find_all_gstins(lines: list[str]) -> list[str]:
    """Return all unique GSTINs found anywhere in the document, in order."""
    return search_pattern(lines, _GSTIN_RE)


def _gstins_in_window(lines: list[str], start: int, window: int) -> list[str]:
    """Extract all GSTINs from lines[start : start + window]."""
    seen: set[str] = set()
    results: list[str] = []
    for line in lines[start: start + window]:
        for m in _GSTIN_RE.finditer(line):
            val = m.group(1)
            if val not in seen:
                seen.add(val)
                results.append(val)
    return results


def _resolve_gstin_by_context(
    lines: list[str],
    keywords: list[str],
    all_gstins: list[str],
    *,
    exclude: Optional[str] = None,
    window: int = _WINDOW_GSTIN,
) -> _ExtractionResult:
    """
    Find a GSTIN by locating a keyword block then scanning ahead.

    Improvement over v1: scans ALL keyword hits and ALL candidates,
    then picks the one with the lowest distance to its keyword.
    """
    sorted_kws = sorted(keywords, key=len, reverse=True)

    best: Optional[_Candidate] = None

    for i, line in enumerate(lines):
        if not any(kw in line.lower() for kw in sorted_kws):
            continue

        for offset in range(0, min(window, len(lines) - i)):
            candidate_line = lines[i + offset]
            for m in _GSTIN_RE.finditer(candidate_line):
                gstin = m.group(1)
                if gstin == exclude:
                    continue
                score = max(0.55, 0.92 - offset * 0.04)
                c = _Candidate(
                    value=gstin,
                    score=score,
                    keyword=line,
                    offset=offset,
                    source_line=candidate_line,
                    method=f'gstin-context[+{offset}] line {i}',
                )
                if best is None or c.score > best.score:
                    best = c

    if best:
        return _ExtractionResult(
            value=best.value,
            confidence=best.score,
            method=best.method,
        )

    return _ExtractionResult()


# ---------------------------------------------------------------------------
# Field-specific resolvers
# ---------------------------------------------------------------------------

def resolve_supplier_gstin(
    lines: list[str],
    all_gstins: list[str],
    buyer_gstin: Optional[str] = None,
) -> _ExtractionResult:
    """
    Identify the supplier (seller/vendor) GSTIN.

    Strategy
    --------
    1. Context-window scan using supplier keywords.
    2. If only two GSTINs exist and buyer is already known, the remaining
       one is the supplier.
    3. Returns None rather than guessing when context is ambiguous.
    """
    result = _resolve_gstin_by_context(
        lines, _SUPPLIER_KEYWORDS, all_gstins, exclude=buyer_gstin
    )
    if result.value:
        return result

    if buyer_gstin and len(all_gstins) == 2:
        remaining = [g for g in all_gstins if g != buyer_gstin]
        if remaining:
            return _ExtractionResult(
                value=remaining[0],
                confidence=0.60,
                method='deduction (two GSTINs, buyer known)',
            )

    logger.debug('resolve_supplier_gstin: no supplier GSTIN identified')
    return _ExtractionResult()


def resolve_buyer_gstin(
    lines: list[str],
    all_gstins: list[str],
    supplier_gstin: Optional[str] = None,
) -> _ExtractionResult:
    """
    Identify the buyer (recipient/beneficiary) GSTIN.

    Strategy mirrors resolve_supplier_gstin but uses buyer keywords.
    Uses a wider search window because buyer GSTIN often appears several
    lines below "Company Name" or "GSTN Number".
    """
    result = _resolve_gstin_by_context(
        lines, _BUYER_KEYWORDS, all_gstins, exclude=supplier_gstin
    )
    if result.value:
        return result

    if supplier_gstin and len(all_gstins) == 2:
        remaining = [g for g in all_gstins if g != supplier_gstin]
        if remaining:
            return _ExtractionResult(
                value=remaining[0],
                confidence=0.60,
                method='deduction (two GSTINs, supplier known)',
            )

    logger.debug('resolve_buyer_gstin: no buyer GSTIN identified')
    return _ExtractionResult()


# ---------------------------------------------------------------------------
# Free-text scorers (names, payment terms, bank name/branch)
# ---------------------------------------------------------------------------

def _score_free_text(val: str, source_line: str, offset: int) -> float:
    """
    Generic scorer for short free-text fields (payment terms, bank name,
    branch, ...).

    Evaluates the label-stripped value (see `_strip_label_prefix`), since
    a window-scanned candidate line may embed a different label than the
    anchor keyword that found it (e.g. "Guest Name : Ms. Arpita
    Mukherjee" scanned from a "Recipient" anchor). Rejects candidates
    that are empty, too short/long, purely numeric, or that are
    themselves a GSTIN. Otherwise scores based on plausible text shape
    (contains letters, more than one word).
    """
    val = _strip_label_prefix(val.strip())

    if not val or len(val) < 2 or len(val) > 80:
        return 0.0
    if not re.search(r'[A-Za-z]', val):
        return 0.0
    if val.isdigit():
        return 0.0
    if _GSTIN_RE.search(val.upper()):
        return 0.0
    if not re.search(r'\s', val) and len(val) > 20:
        # A single unbroken token longer than 20 chars is almost always a
        # code/hash/IRN, not a real name/term (e.g. a 64-char IRN hex
        # string), even though it technically "contains letters".
        return 0.0

    score = 0.55
    if len(val.split()) >= 2:
        score += 0.10

    return min(1.0, score)


def _score_company_name(val: str, source_line: str, offset: int) -> float:
    """
    Scorer for vendor/buyer display names.

    Builds on `_score_free_text` (which already strips an embedded label
    prefix) and additionally:
    - Rejects candidates that are just the keyword/label itself
      (e.g. "Supplier" with nothing else on the line).
    - Rejects lines that are actually a GSTIN, or a GSTIN label, under
      any common spelling ("GSTIN", "GSTN", "GST No", ...).
    - Rewards common company-name suffixes (Pvt, Ltd, LLP, Hotel, ...).
    """
    val = _strip_label_prefix(val.strip())
    base = _score_free_text(val, source_line, offset)
    if base <= 0.0:
        return base

    lower = val.lower().strip(' :.-')

    if lower in _NAME_LABEL_REJECT_WORDS:
        return 0.0
    if _GSTIN_RE.search(val.upper()):
        return 0.0
    if re.search(r'\bgst\w*\b', lower):
        # Covers "gstin", "gstn", "gst no", "gst number", etc. -- these
        # are GSTIN labels, not company names, regardless of OCR spelling.
        return 0.0

    score = base
    if re.search(
        r'\b(pvt|ltd|llp|inc|limited|corp|corporation|company|co\.?|'
        r'hotel|resorts?|enterprises?|industries|group|associates)\b',
        lower,
    ):
        score += 0.20
    if len(val.split()) >= 2:
        score += 0.05

    return min(1.0, score)


def resolve_vendor_name(lines: list[str]) -> _ExtractionResult:
    """
    Extract the vendor / supplier display name (as opposed to their GSTIN).

    Anchors on the same keywords used for supplier GSTIN resolution
    ("supplier", "vendor", "sold by", ...) and scores nearby lines for
    how company-name-shaped they look.
    """
    result = _extract_best_candidate(
        lines,
        _SUPPLIER_KEYWORDS,
        _FULL_LINE_RE,
        _score_company_name,
        window=_WINDOW_NAME,
        field_name='Vendor Name',
    )
    return _finalize_name_result(result)


def resolve_buyer_name(lines: list[str]) -> _ExtractionResult:
    """
    Extract the buyer / recipient display name (as opposed to their GSTIN).

    Anchors on the same keywords used for buyer GSTIN resolution
    ("bill to", "guest name", "company name", ...).
    """
    result = _extract_best_candidate(
        lines,
        _BUYER_KEYWORDS,
        _FULL_LINE_RE,
        _score_company_name,
        window=_WINDOW_NAME,
        field_name='Buyer Name',
    )
    return _finalize_name_result(result)


# ---------------------------------------------------------------------------
# Invoice / PO number scorer
# ---------------------------------------------------------------------------

# Patterns that indicate a value is NOT an invoice/PO number
_ROOM_RE = re.compile(r'^\d{1,4}$')           # pure short integer: room / floor
_ALL_ALPHA_SHORT = re.compile(r'^[A-Za-z]{1,4}$')  # plan codes: CP, EP, B2B

def _score_alnum_code(val: str, source_line: str, offset: int) -> float:
    """
    Score a candidate alphanumeric reference code.

    Used for both invoice numbers and PO numbers, since both are short
    alphanumeric identifiers with the same shape.

    Rules
    -----
    - Must contain at least one digit                      → hard requirement
    - Must be >= 4 characters                              → hard requirement
    - Must not be a reject word (B2B, CP, Tax, …)          → hard requirement
    - Pure short digits (1–4 digits only) → room/floor nr  → reject
    - Longer mixed alphanumeric → good candidate
    - Contains slash or dash → typical invoice/PO format bonus
    - Presence of OCR artefacts like leading '+' → strip & penalise
    """
    val = val.strip().lstrip('+').strip()

    # Hard requirements
    if not re.search(r'\d', val):
        return 0.0
    if len(val) < 4:
        return 0.0
    if val.lower() in _INVOICE_NO_REJECT_WORDS:
        return 0.0
    if _ROOM_RE.match(val):
        return 0.0
    if _ALL_ALPHA_SHORT.match(val):
        return 0.0

    # Penalise values that look like dates
    for pat, _ in _DATE_PATTERNS:
        if pat.fullmatch(val):
            return 0.0

    score = 0.70  # base

    # Bonus: contains a letter + digit mix (classic invoice/PO number format)
    if re.search(r'[A-Za-z]', val) and re.search(r'\d', val):
        score += 0.15

    # Bonus: contains separator characters common in invoice/PO numbers
    if re.search(r'[/\-_]', val):
        score += 0.05

    # Bonus: longer values are more likely to be real reference numbers
    if len(val) >= 8:
        score += 0.05

    # Penalise very long values (might be an IRN or address fragment)
    if len(val) > 35:
        score -= 0.30

    return min(1.0, score)


def resolve_invoice_number(lines: list[str]) -> _ExtractionResult:
    """
    Extract the invoice number.

    Handles layouts:
    - "Invoice No: INV123"                    (inline)
    - "Invoice No\\nINV123"                   (next line)
    - "Invoice No : IBIS-73262 Document Date : 2025-10-11 Room Number : 318"
      (multi-field single line – inline-after-colon isolates the right token)
    - "Category : B2B    Invoice No : 372502  Document Date : 25/09/2025"
      (all on one line)

    OCR artefact cleaning
    ---------------------
    Strips leading '+' characters that Tesseract sometimes prepends.
    """
    result = _extract_best_candidate(
        lines,
        _INVOICE_NO_KEYWORDS,
        _ALNUM_CODE_RE,
        _score_alnum_code,
        window=_WINDOW_INVOICE_NO,
        avoid_keywords=_INVOICE_NO_AVOID_KEYWORDS,
        field_name='Invoice Number',
    )

    if result.value:
        # Clean OCR artefact: strip leading '+'
        result.value = result.value.lstrip('+').strip()
        return result

    return _ExtractionResult()


def resolve_po_number(lines: list[str]) -> _ExtractionResult:
    """
    Extract the purchase-order (PO) number, if present on the invoice.

    Uses the same alphanumeric-code scorer as resolve_invoice_number since
    PO numbers share the same general shape (short mixed alphanumeric
    tokens, often with slashes/dashes).
    """
    result = _extract_best_candidate(
        lines,
        _PO_NUMBER_KEYWORDS,
        _ALNUM_CODE_RE,
        _score_alnum_code,
        window=_WINDOW_PO,
        field_name='PO Number',
    )

    if result.value:
        result.value = result.value.lstrip('+').strip()
        return result

    return _ExtractionResult()


# ---------------------------------------------------------------------------
# Payment terms
# ---------------------------------------------------------------------------

def resolve_payment_terms(lines: list[str]) -> _ExtractionResult:
    """
    Extract payment terms (e.g. "Net 30", "Due on receipt", "Advance
    payment"). Free-text field, so this reuses the generic free-text
    scorer rather than a fixed-shape regex.
    """
    result = _extract_best_candidate(
        lines,
        _PAYMENT_TERMS_KEYWORDS,
        _FULL_LINE_RE,
        _score_free_text,
        window=_WINDOW_PAYMENT_TERMS,
        field_name='Payment Terms',
    )
    return _finalize_free_text_result(result)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_date_string(raw: str) -> Optional[str]:
    """
    Parse a raw date string into ISO-8601 (YYYY-MM-DD).
    Returns None if parsing fails.
    """
    raw = raw.strip()

    for pattern, fmt in _DATE_PATTERNS:
        m = pattern.search(raw)
        if not m:
            continue

        try:
            if fmt == 'dmy4':
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif fmt == 'ymd4':
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif fmt == 'dmy2':
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                y += 2000 if y < 50 else 1900
            elif fmt == 'dMonY':
                d = int(m.group(1))
                mo = _MONTH_MAP.get(m.group(2).lower()[:3], 0)
                y = int(m.group(3))
                if y < 100:
                    y += 2000 if y < 50 else 1900
            else:
                continue

            return datetime(y, mo, d).strftime('%Y-%m-%d')

        except (ValueError, KeyError):
            continue

    return None


def _score_date(val: str, source_line: str, offset: int) -> float:
    """
    Score a date candidate.

    A value scores well only if it is actually parseable as a valid date.
    Arrival / departure dates are rejected via avoid_keywords at the caller.
    """
    parsed = _parse_date_string(val)
    if not parsed:
        return 0.0

    # Prefer recent dates (2020–2030) as invoice dates
    try:
        dt = datetime.strptime(parsed, '%Y-%m-%d')
        if 2015 <= dt.year <= 2035:
            score = 0.85
        else:
            score = 0.50  # very old or far-future date is suspicious
    except ValueError:
        score = 0.70

    return score


def resolve_invoice_date(lines: list[str]) -> _ExtractionResult:
    """
    Extract the invoice date, normalised to YYYY-MM-DD.

    Handles:
    - "Invoice Date: 30/09/2025"
    - "Invoice Date\\n30/09/2025"
    - "Document Date : 2025-09-30"
    - "Bill Date: 01-Jan-25"
    - "Invoice No : 372502  Document Date : 25/09/2025" (multi-field line)

    Arrival / departure / check-in / check-out dates are excluded via
    avoid_keywords so we never return a hotel stay date as the invoice date.
    """
    _ANY_DATE_RE = re.compile(
        r'('
        r'\d{1,2}[/-]\d{2}[/-]\d{2,4}'
        r'|\d{4}[/-]\d{2}[/-]\d{2}'
        r'|\d{1,2}[- ]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[- ]+\d{2,4}'
        r')',
        re.IGNORECASE,
    )

    result = _extract_best_candidate(
        lines,
        _INVOICE_DATE_KEYWORDS,
        _ANY_DATE_RE,
        _score_date,
        window=_WINDOW_INVOICE_DATE,
        avoid_keywords=_DATE_AVOID_KEYWORDS,
        field_name='Invoice Date',
    )

    if result.value:
        parsed = _parse_date_string(result.value)
        if parsed:
            return _ExtractionResult(
                value=parsed,
                confidence=result.confidence,
                method=result.method,
            )

    logger.debug('resolve_invoice_date: no date extracted')
    return _ExtractionResult()


# ---------------------------------------------------------------------------
# IRN resolver (with space-collapse for OCR line-wrap)
# ---------------------------------------------------------------------------

def _collapse_hex_fragments(lines: list[str], start: int, window: int) -> Optional[str]:
    """
    OCR sometimes splits a 64-char hex string across multiple lines or
    inserts spaces.  This function:
    1. Collects text from start..(start+window) lines.
    2. Strips all whitespace to produce a single string.
    3. Searches for any 64-char hex substring.

    Returns the first 64-char hex found, or None.
    """
    fragment = ''.join(lines[start: start + window])
    # Remove spaces
    compact = re.sub(r'\s+', '', fragment)
    m = re.search(r'[0-9a-fA-F]{64}', compact)
    return m.group(0) if m else None


def resolve_irn(lines: list[str]) -> _ExtractionResult:
    """
    Extract the IRN (Invoice Reference Number) – a 64-character hex string.

    Improvements over v1
    --------------------
    - Collapses spaces within the context window to handle OCR line-wrap.
    - Scans all IRN-label positions and picks the nearest valid hex.
    - Falls back to first unlabelled 64-hex in the document.
    """
    # First try: direct match of solid 64-char hex anywhere
    all_irns = search_pattern(lines, _IRN_RE)

    irn_label_indices = _keyword_line_indices(lines, ['irn', 'invoice reference'])

    if not irn_label_indices:
        if all_irns:
            return _ExtractionResult(
                value=all_irns[0],
                confidence=0.65,
                method='unlabelled 64-char hex',
            )
        return _ExtractionResult()

    best: Optional[_ExtractionResult] = None

    for label_idx in irn_label_indices:
        scan_window = _WINDOW_IRN

        # Strategy 1: look for solid hex in window
        for offset in range(0, min(scan_window, len(lines) - label_idx)):
            m = _IRN_RE.search(lines[label_idx + offset])
            if m:
                conf = max(0.60, 0.95 - offset * 0.05)
                candidate = _ExtractionResult(
                    value=m.group(1),
                    confidence=conf,
                    method=f'IRN label context[+{offset}]',
                )
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
                break

        # Strategy 2: collapse whitespace and search for fragmented hex
        if best is None:
            collapsed = _collapse_hex_fragments(lines, label_idx, scan_window)
            if collapsed:
                best = _ExtractionResult(
                    value=collapsed,
                    confidence=0.75,
                    method='IRN collapsed-whitespace match',
                )

    if best:
        return best

    # Final fallback: first unlabelled 64-hex in document
    if all_irns:
        return _ExtractionResult(
            value=all_irns[0],
            confidence=0.65,
            method='fallback unlabelled',
        )

    return _ExtractionResult()


# ---------------------------------------------------------------------------
# Amount scorers and resolvers (total, subtotal, CGST, SGST, IGST)
# ---------------------------------------------------------------------------

def _score_amount(val: str, source_line: str, offset: int) -> float:
    """
    Score a currency amount candidate for the GRAND TOTAL field.
    Prefers larger amounts (more likely to be the grand total than a line item).
    """
    try:
        numeric = float(val.replace(',', ''))
    except ValueError:
        return 0.0

    if numeric <= 0:
        return 0.0

    # Reasonable invoice total range (not a GST rate, not a room number)
    if numeric < 10:
        return 0.10  # probably a GST rate percentage
    if numeric < 100:
        return 0.40

    score = 0.75

    # Larger amounts score higher (grand total > line item)
    if numeric >= 1000:
        score += 0.10
    if numeric >= 10000:
        score += 0.05

    return min(1.0, score)


def _score_tax_component(val: str, source_line: str, offset: int) -> float:
    """
    Score a currency amount candidate for a tax/subtotal component
    (subtotal, CGST, SGST, IGST).

    Unlike the grand total, these are legitimately often small numbers
    (a few hundred rupees or less), so small values are NOT penalised the
    way they are in `_score_amount`.
    """
    try:
        numeric = float(val.replace(',', ''))
    except ValueError:
        return 0.0

    if numeric < 0:
        return 0.0
    if numeric == 0:
        # A tax component can legitimately be zero (e.g. IGST on an
        # intra-state invoice), but that's a weaker signal than a real
        # non-zero amount actually found near the keyword.
        return 0.30

    score = 0.75
    if numeric >= 10:
        score += 0.05
    if numeric >= 100:
        score += 0.05

    return min(1.0, score)


def resolve_total_amount(lines: list[str]) -> _ExtractionResult:
    """
    Extract the grand total / invoice value.

    Handles:
    - "Grand Total: Rs.1,23,456.00"
    - "Invoice Value\\n1,23,456"
    - "Net Amount  1,23,456.00"
    - "Total Inv. Value   11,800.00" (EzyInvoice layout)
    """
    result = _extract_best_candidate(
        lines,
        _TOTAL_KEYWORDS,
        _AMOUNT_RE,
        _score_amount,
        window=_WINDOW_TOTAL,
        field_name='Total Amount',
    )

    if result.value:
        result.value = result.value.replace(',', '').strip()
        return result

    return _ExtractionResult()


def _resolve_amount_field(
    lines: list[str],
    keywords: list[str],
    window: int,
    field_name: str,
) -> _ExtractionResult:
    """
    Shared implementation for subtotal / CGST / SGST / IGST resolution.
    All four are "find a currency amount near this keyword" fields that
    differ only in their keyword list.
    """
    result = _extract_best_candidate(
        lines,
        keywords,
        _AMOUNT_RE,
        _score_tax_component,
        window=window,
        field_name=field_name,
    )

    if result.value:
        result.value = result.value.replace(',', '').strip()
        return result

    return _ExtractionResult()


def resolve_subtotal(lines: list[str]) -> _ExtractionResult:
    """Extract the pre-tax / taxable value."""
    return _resolve_amount_field(lines, _SUBTOTAL_KEYWORDS, _WINDOW_TAX, 'Subtotal')


def resolve_cgst(lines: list[str]) -> _ExtractionResult:
    """Extract the Central GST amount."""
    return _resolve_amount_field(lines, _CGST_KEYWORDS, _WINDOW_TAX, 'CGST')


def resolve_sgst(lines: list[str]) -> _ExtractionResult:
    """Extract the State GST amount."""
    return _resolve_amount_field(lines, _SGST_KEYWORDS, _WINDOW_TAX, 'SGST')


def resolve_igst(lines: list[str]) -> _ExtractionResult:
    """Extract the Integrated GST amount."""
    return _resolve_amount_field(lines, _IGST_KEYWORDS, _WINDOW_TAX, 'IGST')


# ---------------------------------------------------------------------------
# Currency resolver
# ---------------------------------------------------------------------------

def resolve_currency(lines: list[str], *, has_gstin: bool = False) -> _ExtractionResult:
    """
    Detect the invoice currency.

    Strategy
    --------
    1. Scan every line for currency symbols/codes (₹/Rs/INR, $/USD, €/EUR,
       £/GBP) and tally hits per code; the most frequent code wins.
    2. If no explicit currency marker is found anywhere but the document
       contains at least one Indian GSTIN, default to INR at moderate
       confidence -- a GSTIN is a strong signal this is an Indian GST
       invoice.
    3. Otherwise returns None rather than guessing.
    """
    counts: dict[str, int] = {}
    for line in lines:
        for pattern, code in _CURRENCY_PATTERNS:
            if pattern.search(line):
                counts[code] = counts.get(code, 0) + 1

    if counts:
        best_code = max(counts, key=lambda c: counts[c])
        confidence = 0.80 if counts[best_code] >= 2 else 0.65
        return _ExtractionResult(
            value=best_code,
            confidence=confidence,
            method='currency-symbol-scan',
        )

    if has_gstin:
        return _ExtractionResult(
            value='INR',
            confidence=0.55,
            method='defaulted (GSTIN present)',
        )

    return _ExtractionResult()


# ---------------------------------------------------------------------------
# Bank details resolver
# ---------------------------------------------------------------------------

def _score_account_number(val: str, source_line: str, offset: int) -> float:
    """
    Score a candidate bank account number.
    Plausible Indian bank account numbers are 9-18 digits; anything outside
    8-20 digits is rejected outright by the 8-20 digit regex already, but
    we still prefer the "typical" middle of that range.
    """
    if not val.isdigit():
        return 0.0

    length = len(val)
    if length < 8 or length > 20:
        return 0.0

    score = 0.70
    if 9 <= length <= 18:
        score += 0.15

    return min(1.0, score)


def resolve_bank_details(lines: list[str]) -> tuple[dict, float]:
    """
    Extract bank payment details as a nested dict.

    Looks for an account number, IFSC code, bank name, and branch
    independently (each may appear in a different part of the document),
    then assembles them into a single dict. Missing sub-fields are `None`
    rather than omitted, so downstream consumers always see the same keys.

    Returns
    -------
    tuple[dict, float]
        (bank_details dict, aggregate confidence for the "bank_details" field
        -- the highest confidence among whichever sub-fields were found,
        or 0.0 if nothing was found at all).
    """
    account_result = _extract_best_candidate(
        lines,
        _BANK_ACCOUNT_KEYWORDS,
        _ACCOUNT_NO_RE,
        _score_account_number,
        window=_WINDOW_BANK,
        field_name='Bank Account No',
    )

    ifsc_matches = search_pattern(lines, _IFSC_RE)
    ifsc_value = ifsc_matches[0] if ifsc_matches else None

    bank_name_result = _finalize_free_text_result(_extract_best_candidate(
        lines,
        _BANK_NAME_KEYWORDS,
        _FULL_LINE_RE,
        _score_free_text,
        window=3,
        field_name='Bank Name',
    ))

    branch_result = _finalize_free_text_result(_extract_best_candidate(
        lines,
        _BANK_BRANCH_KEYWORDS,
        _FULL_LINE_RE,
        _score_free_text,
        window=3,
        field_name='Branch',
    ))

    details = {
        'account_number': account_result.value,
        'ifsc_code': ifsc_value,
        'bank_name': bank_name_result.value,
        'branch': branch_result.value,
    }

    found_confidences = [
        c for c in (
            account_result.confidence,
            0.85 if ifsc_value else 0.0,
            bank_name_result.confidence,
            branch_result.confidence,
        )
        if c > 0
    ]
    aggregate_confidence = max(found_confidences) if found_confidences else 0.0

    return details, aggregate_confidence


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def resolve_fields(raw_ocr_text: str) -> dict:
    """
    Main entry point.  Accepts raw OCR text and returns a fully structured
    extraction result dictionary.

    Parameters
    ----------
    raw_ocr_text : str
        The raw string from Tesseract / PaddleOCR.

    Returns
    -------
    dict with keys:
        supplier_gstin, buyer_gstin, vendor_name, buyer_name, invoice_no,
        invoice_date, total_amount, subtotal, cgst, sgst, igst, currency,
        po_number, payment_terms, bank_details, irn, gstins_found,
        confidence
    """
    lines = preprocess_text(raw_ocr_text)

    if not lines:
        logger.warning('resolve_fields: received empty text')
        return ResolvedFields().to_dict()

    # --- Step 1: collect all GSTINs globally ---
    all_gstins = find_all_gstins(lines)
    logger.debug('GSTINs found: %s', all_gstins)

    # --- Step 2: supplier GSTIN (resolve first to enable exclusion in buyer) ---
    supplier_result = resolve_supplier_gstin(lines, all_gstins)

    # --- Step 3: buyer GSTIN (exclude supplier to prevent collision) ---
    buyer_result = resolve_buyer_gstin(
        lines, all_gstins, supplier_gstin=supplier_result.value
    )

    # --- Retry supplier excluding buyer (handles ambiguous document order) ---
    if not supplier_result.value and buyer_result.value:
        supplier_result = resolve_supplier_gstin(
            lines, all_gstins, buyer_gstin=buyer_result.value
        )

    # --- Step 4: vendor / buyer display names ---
    vendor_name_result = resolve_vendor_name(lines)
    buyer_name_result = resolve_buyer_name(lines)

    # --- Step 5: invoice number ---
    inv_no_result = resolve_invoice_number(lines)

    # --- Step 6: invoice date ---
    inv_date_result = resolve_invoice_date(lines)

    # --- Step 7: IRN ---
    irn_result = resolve_irn(lines)

    # --- Step 8: amounts (total, subtotal, CGST, SGST, IGST) ---
    total_result = resolve_total_amount(lines)
    subtotal_result = resolve_subtotal(lines)
    cgst_result = resolve_cgst(lines)
    sgst_result = resolve_sgst(lines)
    igst_result = resolve_igst(lines)

    # --- Step 9: currency ---
    currency_result = resolve_currency(lines, has_gstin=bool(all_gstins))

    # --- Step 10: PO number ---
    po_result = resolve_po_number(lines)

    # --- Step 11: payment terms ---
    payment_terms_result = resolve_payment_terms(lines)

    # --- Step 12: bank details ---
    bank_details, bank_details_confidence = resolve_bank_details(lines)

    # --- Assemble output ---
    output = ResolvedFields(
        supplier_gstin=supplier_result.value,
        buyer_gstin=buyer_result.value,
        vendor_name=vendor_name_result.value,
        buyer_name=buyer_name_result.value,
        invoice_no=inv_no_result.value,
        invoice_date=inv_date_result.value,
        total_amount=total_result.value,
        subtotal=subtotal_result.value,
        cgst=cgst_result.value,
        sgst=sgst_result.value,
        igst=igst_result.value,
        currency=currency_result.value,
        po_number=po_result.value,
        payment_terms=payment_terms_result.value,
        bank_details=bank_details,
        irn=irn_result.value,
        gstins_found=all_gstins,
        confidence={
            'supplier_gstin':  supplier_result.confidence,
            'buyer_gstin':     buyer_result.confidence,
            'vendor_name':     vendor_name_result.confidence,
            'buyer_name':      buyer_name_result.confidence,
            'invoice_no':      inv_no_result.confidence,
            'invoice_date':    inv_date_result.confidence,
            'total_amount':    total_result.confidence,
            'subtotal':        subtotal_result.confidence,
            'cgst':            cgst_result.confidence,
            'sgst':            sgst_result.confidence,
            'igst':            igst_result.confidence,
            'currency':        currency_result.confidence,
            'po_number':       po_result.confidence,
            'payment_terms':   payment_terms_result.confidence,
            'bank_details':    bank_details_confidence,
            'irn':             irn_result.confidence,
        },
    )

    logger.info(
        'resolve_fields complete | invoice_no=%s date=%s supplier=%s buyer=%s total=%s',
        output.invoice_no,
        output.invoice_date,
        output.supplier_gstin,
        output.buyer_gstin,
        output.total_amount,
    )

    return output.to_dict()


# ---------------------------------------------------------------------------
# Quick smoke-test  (python field_resolver.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json
    logging.basicConfig(level=logging.DEBUG)

    # ---- Test 1: standard layout ----
    SAMPLE_STANDARD = """
    Tax Invoice

    Invoice No : INV-2025-001
    Document Date : 2025-09-30
    PO Number : PO-88213
    Payment Terms : Net 30

    Supplier
    ABC Exports Pvt Ltd
    GSTIN: 29ABCDE1234F1Z5

    Bill To
    XYZ Traders
    GSTIN: 27XYZPQ9876K1Z3

    IRN:
    Ack No: 1234567890
    Ack Date: 30/09/2025
    a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6aabb

    Taxable Amount: Rs.1,00,000.00
    CGST: Rs.11,728.00
    SGST: Rs.11,728.00
    Grand Total: Rs.1,23,456.00

    Bank Name: HDFC Bank
    Account No: 123456789012
    IFSC: HDFC0001234
    Branch: MG Road Branch
    """

    # ---- Test 2: multi-column layout (the hard case) ----
    SAMPLE_MULTICOLUMN = """
    Tax Invoice

    Category : B2B     Invoice No : IBIS-73262    Document Date : 2025-10-11    Room Number : 318
    Document Type : Tax Invoice    Confirmation No : 607917070

    Supplier
    GSTIN : 06AABCI2732H1ZW
    IBIS Gurgaon Golf Course Road IT

    Recipient
    Guest Name : Ms. Arpita Mukherjee
    GSTIN: 06AAEFE1778R1ZU

    IRN : d6f43f55918067e36569c11f7b88e6bfccd08fdb82e4deef65a6f7ef7f17050e

    Grand Total: 27615.42
    """

    # ---- Test 3: Ramada layout (buyer GSTIN after "GSTN Number" label) ----
    SAMPLE_RAMADA = """
    TAX INVOICE

    Guest Name : MR MEHTA CHETAN
    Company Name : YATRA FOR BUSINESS PRIVATE LIMITED
    GSTN Number : 07AAEFE1763C1ZU
    Company Address : 3rd Floor, Unit No. 1, Vasant Arcade

    Invoice Date : 30/09/25
    Tax Invoice No. : F2551BIL26007081

    Property GSTN# : 06AADCG1506B1ZE

    IRN NO: ed7b42a4723f93c66e24fd990a96c1c5a4a1bc27dd96c023ecc1d68624f78f45

    Net Amount: 11819.80
    """

    # ---- Test 4: hotel label-block layout (labels first, values below) ----
    # This was the failing case: all labels appear before line 10, all values
    # appear after line 10.  Both invoice_no and invoice_date were missed
    # because the context window was too small.
    SAMPLE_HOTEL_LABELBLOCK = """
Invoice Number
Invoice Date
Room No

Room Type
Reservation #
Number of Pax
Arrival Date
Departure Date
Plan

Billing Instruction
Tariff

+ FM0636BIL0004439

1 18/10/25
1417
:STD

: 119811
21
17/10/25
18/10/25
"""

    for label, sample in [
        ('STANDARD', SAMPLE_STANDARD),
        ('MULTI-COLUMN', SAMPLE_MULTICOLUMN),
        ('RAMADA', SAMPLE_RAMADA),
        ('HOTEL LABEL-BLOCK', SAMPLE_HOTEL_LABELBLOCK),
    ]:
        print(f'\n{"="*60}')
        print(f'TEST: {label}')
        print('='*60)
        result = resolve_fields(sample)
        for k, v in result.items():
            if k != 'confidence':
                print(f'  {k:<20}: {v}')
        print('  confidence:')
        for k, v in result['confidence'].items():
            print(f'    {k:<18}: {v:.2f}')