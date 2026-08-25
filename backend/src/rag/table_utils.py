"""
Structured table utilities.

Parses '|'-delimited chunk text (the format produced by csv_parser.py and
the table-extraction path in pdf_parser.py) into row dictionaries, and
provides filter/aggregate helpers so questions like "which X belong to Y"
or "total of Z for Y" get answered by exact code logic instead of by an
LLM scanning many rows — small local models are unreliable at multi-row
filtering and multi-number arithmetic even when given fully correct
context, so this logic never touches the LLM for the filtering/math part.
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
logger = logging.getLogger(__name__)


@dataclass
class ParsedTable:
    headers: list[str]
    rows: list[dict[str, str]]


def _normalize_header(headers: list[str]) -> tuple[str, ...]:
    """Normalize headers so minor spacing/case/punctuation differences
    across page breaks or chunk boundaries don't break table stitching."""
    return tuple(re.sub(r"\s+", " ", h.strip().lower()) for h in headers)


# Header alias map: maps common user terms to canonical header tokens to help
# natural-language -> header resolution. Keys are normalized tokens that may
# appear in questions; values are normalized substrings to match against
# available headers. Keep this conservative and extensible.
_HEADER_ALIASES: dict[str, str] = {
    "year": "year",
    "years": "year",
    "country": "country",
    "countries": "country",
    "segment": "segment",
    "segments": "segment",
    "product": "product",
    "products": "product",
    "discount": "discount band",
    "discounts": "discount band",
    "discount band": "discount band",
    "unit": "units sold",
    "units": "units sold",
    "manufacturing price": "manufacturing price",
    "sale price": "sale price",
    "sale": "sale price",
    "sales": "sales",
    "gross": "gross sales",
    "gross sales": "gross sales",
    "profit": "profit",
    "cogs": "cogs",
}

_GENERIC_HEADER_STOPWORDS: set[str] = {
    "system", "type", "data", "value", "values", "name", "names", "id",
    "date", "dates", "status", "description", "number", "numbers", "code",
    "codes", "category", "categories", "item", "items", "table", "column",
    "columns", "row", "rows", "field", "fields", "key", "keys", "label",
    "labels", "info", "information", "details", "detail",
}


def resolve_header(token: str, headers: list[str]) -> str | None:
    """Resolve a user-provided token (like 'country' or 'year') to one of the
    actual header names present in `headers`. Matching is case-insensitive and
    uses the alias map as a first pass. Returns the first matching header or
    None if no good match found.
    """
    if not token or not headers:
        return None
    token_n = _normalize(token)
    # Map token via aliases
    mapped = _HEADER_ALIASES.get(token_n, token_n)
    # Prefer headers that contain the mapped token as a whole-word
    for h in headers:
        if mapped in _normalize(h):
            return h
    # Fallback: try matching any header words
    for h in headers:
        words = set(re.findall(r"\w+", _normalize(h)))
        if mapped in words or token_n in words:
            return h
    return None

def _table_relevance_score(headers: list[str], query: str) -> int:
    """Counts how many non-trivial query words appear in this table's
    headers. Query-agnostic — works for any language/domain as long as
    the query and headers share tokens (section names, column labels,
    entity types). Returns 0 if no query given or no overlap, which
    falls back to pure row-count ranking (old behavior).

    Generic administrative header terms (_GENERIC_HEADER_STOPWORDS) are
    excluded from the overlap count. Without this, a query and an
    unrelated table could "match" purely because both happen to mention
    a word like "system" or "type" — enough on its own to pass
    table_passes_acceptance_gate's header_score >= 1 check and let a
    completely unrelated table through (e.g. a CAP Theorem table with a
    "System" column accepted for a GDPR compliance query)."""
    if not query:
        return 0
    query_words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
    header_words = {w for h in headers for w in re.findall(r"\w+", h.lower())}
    query_words -= _GENERIC_HEADER_STOPWORDS
    header_words -= _GENERIC_HEADER_STOPWORDS
    return len(query_words & header_words)


def table_passes_acceptance_gate(table: ParsedTable, query: str) -> tuple[bool, str]:
    """
    Returns (accepted, reason). Combines three deterministic signals —
    no LLM call, no embeddings (yet):
      - header_token_overlap: existing _table_relevance_score()
      - cell_value_overlap_ratio: % of rows where any query token appears
        in any cell value
      - quoted_entity_present: if query has a quoted phrase, does it
        appear verbatim (normalized) in any cell?
    Accept if ANY of:
      - quoted_entity_present is True
      - header_token_overlap >= 1
      - cell_value_overlap_ratio >= 0.05
    Otherwise reject.
    """
    if table is None or not table.rows:
        return False, "no table rows"

    # header overlap
    header_score = _table_relevance_score(table.headers, query)

    # quoted entity detection (simple quoted-phrase extractor)
    quoted_matches = re.findall(r"['\"]([^'\"]{2,80})['\"]", query)
    quoted_entity_present = False
    if quoted_matches:
        quoted = quoted_matches[0].strip()
        qnorm = _normalize(quoted)
        for row in table.rows:
            for v in row.values():
                if qnorm and qnorm in _normalize(v):
                    quoted_entity_present = True
                    break
            if quoted_entity_present:
                break

    # cell value overlap ratio: fraction of rows containing any query token
    query_tokens = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
    if not query_tokens:
        cell_value_overlap_ratio = 0.0
    else:
        hit_rows = 0
        total = max(len(table.rows), 1)
        for row in table.rows:
            found = False
            for v in row.values():
                val_norm = _normalize(str(v))
                for t in query_tokens:
                    if t in val_norm:
                        found = True
                        break
                if found:
                    break
            if found:
                hit_rows += 1
        cell_value_overlap_ratio = hit_rows / total

    # Acceptance rules
    if quoted_entity_present:
        return True, f"quoted_entity_present (matches '{quoted}')"
    if header_score >= 1:
        return True, f"header_token_overlap={header_score}"
    if cell_value_overlap_ratio >= 0.05:
        return True, f"cell_value_overlap_ratio={cell_value_overlap_ratio:.3f}"

    return False, (
        f"rejected: header_score={header_score}, "
        f"cell_value_overlap_ratio={cell_value_overlap_ratio:.3f}, "
        f"quoted_present={bool(quoted_matches)}"
    )

def extremum_row(sheet, column: str, mode: str = "max") -> dict | None:
    """Return the raw row dict where `column` is max/min. Uses typed_rows
    for reliable numeric comparison when available."""
    def _num(v):
        try:
            return float(str(v).replace(",", ""))
        except (ValueError, TypeError):
            return None

    if _is_sheet_dict(sheet) and sheet.get("typed_rows"):
        typed = sheet.get("typed_rows", [])
        raw = sheet.get("rows", [])
        pairs = [(t.get(column), r) for t, r in zip(typed, raw)]
    elif _is_sheet_dict(sheet):
        raw = sheet.get("rows", [])
        pairs = [(r.get(column), r) for r in raw]
    elif isinstance(sheet, ParsedTable):
        pairs = [(r.get(column), r) for r in sheet.rows]
    else:
        return None

    best_val, best_row = None, None
    for val, row in pairs:
        n = _num(val)
        if n is None:
            continue
        if best_val is None or (mode == "max" and n > best_val) or (mode == "min" and n < best_val):
            best_val, best_row = n, row
    return best_row

def parse_and_merge_tables(chunks: list[str], query: str = "") -> ParsedTable | None:
    """
    Parse every chunk as a table and stitch same-schema tables together
    into one continuous dataset, using NORMALIZED header comparison
    (case/whitespace-insensitive) instead of exact string equality.

    Selects the table group most RELEVANT to `query` (by header/query word
    overlap) rather than blindly picking the group with the most rows.
    Row count is only used as a tiebreaker (or as the sole criterion if
    query is empty or matches no table's headers) — this keeps behavior
    correct for documents with multiple structurally distinct tables,
    where the biggest table is often NOT the one being asked about.
    """
    groups: dict[tuple, ParsedTable] = {}

    for chunk in chunks:
        if not looks_tabular(chunk):
            continue
        parsed = parse_table(chunk)
        if not parsed:
            continue

        key = _normalize_header(parsed.headers)
        if key not in groups:
            groups[key] = ParsedTable(headers=parsed.headers, rows=list(parsed.rows))
        else:
            groups[key].rows.extend(parsed.rows)

    if not groups:
        return None

    # Evaluate relevance for all parsed table groups
    scored_groups = []
    for group in groups.values():
        relevance = _table_relevance_score(group.headers, query)
        scored_groups.append((relevance, len(group.rows), group))

    # Sort primarily by relevance score, secondarily by row count
    scored_groups.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_relevance, _, best = scored_groups[0]

    # ENTERPRISE GUARD: If a query was provided, but the best-matching table 
    # has ZERO relevance score overlap, it means no table actually matched 
    # what the user is asking for. Refuse to return an unrelated table (like 
    # returning an insurance table for a KW/SLA query) to prevent hallucinations.
    if query and best_relevance == 0:
        logger.warning(f"[parse_and_merge_tables] No table headers matched query '{query}' (best relevance is 0). Rejecting table fallback to avoid wrong data.")
        return None

    seen = set()
    unique_rows = []
    for row in best.rows:
        key = tuple(row.items())
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    return ParsedTable(headers=best.headers, rows=unique_rows)
    
def _normalize(s: str) -> str:
    """Lowercase, strip punctuation/quotes, collapse whitespace — so
    'Pour-Over Dripper Set' vs 'Pour Over Dripper Set' or a stray quote
    from entity extraction doesn't break an otherwise-exact match."""
    s = re.sub(r"[\"'’]", "", s)
    s = re.sub(r"[-_]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

@lru_cache(maxsize=512)
def looks_tabular(text: str) -> bool:
    """True if most non-blank lines contain a ' | ' column separator AND
    those lines share a consistent column count — a real table has the
    same number of columns per row, while garbled prose with stray pipes
    (e.g. PDF extraction artifacts) does not."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    delimited_lines = [l for l in lines if " | " in l]
    if len(delimited_lines) / len(lines) <= 0.6:
        return False

    # Require at least 2 delimited lines with the SAME column count to
    # count as tabular — a lone pipe-containing line (or lines with wildly
    # varying column counts) is not a table.
    col_counts = [l.count("|") for l in delimited_lines]
    if len(delimited_lines) < 2:
        return False
    from collections import Counter
    most_common_count, freq = Counter(col_counts).most_common(1)[0]
    if most_common_count < 1 or freq < 2:
        return False
    return True

_KEYED_CELL_RE = re.compile(r'^([^:|]{1,60}):\s?(.*)$')


def _is_keyed_line(line: str) -> bool:
    """True if most cells in this pipe-delimited line look like
    'Header: value' — the self-describing row format from excel_parser.py."""
    cells = [c.strip() for c in line.split("|")]
    if not cells:
        return False
    matched = sum(1 for c in cells if _KEYED_CELL_RE.match(c))
    return matched / len(cells) > 0.6


def _parse_keyed_row(line: str) -> dict[str, str]:
    row = {}
    for cell in line.split("|"):
        m = _KEYED_CELL_RE.match(cell.strip())
        if m:
            row[m.group(1).strip()] = m.group(2).strip()
    return row

@lru_cache(maxsize=512)
def parse_table(text: str) -> ParsedTable | None:
    """
    Parse '|'-delimited text into a header list and row dicts.
    Returns None if the text doesn't look tabular or yields no valid rows.

    Chunks are often MIXED — narrative prose sentences sitting right next
    to a real pipe-delimited table (e.g. a paragraph ending mid-chunk,
    then a table starting a few lines later). Treating lines[0] as the
    header regardless of content meant a prose sentence could become the
    "header", causing every real table row after it to be counted as
    "misaligned" and dropped. Instead: first isolate only the lines that
    actually look pipe-delimited, and only ever pick a header from among
    those — prose lines are ignored entirely, not treated as skipped rows.
    """
    all_lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not all_lines:
        return None

    pipe_lines = [l for l in all_lines if l.count("|") >= 1]
    if len(pipe_lines) < 2:
        return None

    # Self-describing "Header: value" rows (xlsx export format) — each row
    # carries its own headers, so chunk boundaries never break header/data
    # alignment. Handle this before falling back to "line 0 is the header".
    keyed_lines = [l for l in pipe_lines if _is_keyed_line(l)]
    if keyed_lines and len(keyed_lines) / len(pipe_lines) > 0.6:
        headers: list[str] = []
        rows = []
        for line in keyed_lines:
            row = _parse_keyed_row(line)
            if not row:
                continue
            for k in row:
                if k not in headers:
                    headers.append(k)
            rows.append(row)
        if rows:
            return ParsedTable(headers=headers, rows=rows)

    header_idx = 0
    headers = [h.strip() for h in pipe_lines[0].split("|")]
    if sum(1 for h in headers if h) <= 1 and len(pipe_lines) > 1:
        header_idx = 1
        headers = [h.strip() for h in pipe_lines[1].split("|")]

    if any(len(h.split()) > 8 for h in headers if h):
        logger.warning(f"[parse_table] rejected — header field too long to be a "
                        f"real column label (likely a misaligned data row): "
                        f"{[h for h in headers if len(h.split()) > 8]}")
        return None

    # Reject code-like or fragmented headers — e.g. "O(1)", "PART 2",
    # a header that's a mid-word fragment ("Time Complexi"), or one with
    # no alphabetic content at all. Real table headers are short,
    # plausible column names ("Name", "Price", "Status").
    _CODE_PATTERN = re.compile(r'[(){}\[\]<>]|^\d+$|^[A-Z]{1,4}\s*\d+$')
    plausible_headers = [
        h for h in headers
        if h and not _CODE_PATTERN.search(h)
        and re.search(r'[a-zA-Z]{2,}', h)
        and len(h) <= 40
    ]
    if len(plausible_headers) < max(2, int(len(headers) * 0.5)):
        logger.warning(f"[parse_table] rejected — too few plausible column-name "
                        f"headers (likely fragmented/code text mistaken for a "
                        f"table): headers={headers}")
        return None
    rows = []
    skipped = 0
    for line in pipe_lines[header_idx + 1:]:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) != len(headers):
            skipped += 1
            continue
        rows.append(dict(zip(headers, cells)))

    if skipped:
        logger.warning(f"[parse_table] skipped {skipped} misaligned row(s) out of "
                        f"{len(pipe_lines) - header_idx - 1} pipe-delimited line(s) "
                        f"(prose lines already excluded) — likely a genuine PDF "
                        f"table-extraction alignment issue")

    return ParsedTable(headers=headers, rows=rows) if rows else None


def find_matching_rows(table: ParsedTable, entity: str) -> list[dict[str, str]]:
    """
    Find rows matching `entity`.

    Prefers EXACT (whole-cell, case-insensitive) matches first. Only falls
    back to substring matching if no exact match exists anywhere in the
    table. This prevents a shorter entity name from incorrectly swallowing
    rows belonging to a longer entity name that contains it as a substring
    — e.g. a query for "Acme" matching "Acme" rows AND "Acme Corp" rows,
    which are two different companies. This is a general correctness fix:
    any real-world dataset with entities where one name is a prefix/
    substring of another (person names, product SKUs, company names,
    codes) would hit the same bug without this exact-match-first ordering.
    """
    entity_norm = _normalize(entity)
    if not entity_norm:
        return []

    exact_matches = [
        row for row in table.rows
        if any(_normalize(v) == entity_norm for v in row.values())
    ]
    if exact_matches:
        return exact_matches

    # No exact cell match anywhere — fall back to substring matching, which
    # still helps for genuinely partial/descriptive queries (e.g. entity
    # extractor pulling "Sales" to match a "Sales Department" cell).
    # Checked in BOTH directions: entity-in-cell (extractor pulled a
    # shorter/generic term) AND cell-in-entity (extractor pulled a longer
    # phrase than what's actually stored, e.g. "HR department" vs a cell
    # that just says "HR"). Only one direction was checked before, which
    # meant longer extracted entities could never match a shorter cell
    # value — this is what broke the HR/Sales aggregate questions.
    return [
        row for row in table.rows
        if any(
            entity_norm in _normalize(v) or _normalize(v) in entity_norm
            for v in row.values() if v.strip()
        )
    ]


def find_numeric_column(table: ParsedTable, hint: str | None = None) -> str | None:
    """
    Guess which column holds the numeric field to sum — prefers a header
    matching `hint` (e.g. "area", "amount"), else the most numeric column.
    """
    def is_numeric(v: str) -> bool:
        v = v.replace(",", "").strip()
        if not v:
            return False
        # Accounting-style negatives, e.g. "(1234.56)" -> negative number
        if v.startswith("(") and v.endswith(")"):
            v = "-" + v[1:-1]
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", v))

    candidates = []
    for header in table.headers:
        values = [row.get(header, "") for row in table.rows]
        ratio = sum(1 for v in values if is_numeric(v)) / max(len(values), 1)
        if ratio > 0.6:
            candidates.append((header, ratio))

    if not candidates:
        return None

    if hint:
        for header, _ in candidates:
            if hint.lower() in header.lower():
                return header

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def sum_column(rows: list[dict[str, str]], column: str) -> float:
    total = 0.0
    for row in rows:
        raw = row.get(column, "").replace(",", "").strip()
        try:
            total += float(raw)
        except ValueError:
            continue
    return total


# --- New helpers for structured sheet dicts (from excel_parser.parse_excel) ---
def _is_sheet_dict(obj) -> bool:
    return isinstance(obj, dict) and "headers" in obj and ("rows" in obj or "typed_rows" in obj)


def get_headers_from_sheet(sheet) -> list[str]:
    """Return headers from either a ParsedTable or the new sheet dict format.

    Works with the legacy ParsedTable (headers attribute) or the
    excel_parser sheet dict that contains a 'headers' list.
    """
    if isinstance(sheet, ParsedTable):
        return sheet.headers
    if _is_sheet_dict(sheet):
        return list(sheet.get("headers", []))
    return []


def distinct_values(sheet, column: str) -> list:
    """Return distinct values for `column` from typed_rows if available,
    otherwise from raw rows. Values are returned in insertion order.
    """
    seen = []
    if isinstance(sheet, ParsedTable):
        rows = sheet.rows
        for r in rows:
            v = r.get(column)
            if v not in seen:
                seen.append(v)
        return seen

    if _is_sheet_dict(sheet):
        # Prefer typed_rows for canonical types, but fall back to raw rows
        if sheet.get("typed_rows"):
            for r in sheet.get("typed_rows", []):
                v = r.get(column)
                if v not in seen:
                    seen.append(v)
            return seen
        for r in sheet.get("rows", []):
            v = r.get(column)
            if v not in seen:
                seen.append(v)
    return seen


def row_lookup(sheet, match: dict[str, object]) -> list[dict]:
    """Find rows matching all key/value pairs in `match`.

    Exact typed-value equality is preferred when typed_rows exist. Falls
    back to normalized string matching for string cells.
    Returns list of matching rows (typed_rows if available, else raw rows).
    """
    def _matches_typed(row_typed, match_spec):
        for k, v in match_spec.items():
            if k not in row_typed:
                return False
            cell = row_typed.get(k)
            # Exact equality for numbers/dates
            if isinstance(v, (int, float)) or isinstance(cell, (int, float)):
                try:
                    if float(cell) != float(v):
                        return False
                except Exception:
                    return False
            else:
                if _normalize(str(cell or "")) != _normalize(str(v or "")):
                    return False
        return True

    def _matches_raw(row_raw, match_spec):
        for k, v in match_spec.items():
            if k not in row_raw:
                return False
            if _normalize(str(row_raw.get(k, ""))) != _normalize(str(v or "")):
                return False
        return True

    results = []
    if isinstance(sheet, ParsedTable):
        for r in sheet.rows:
            if _matches_raw(r, match):
                results.append(r)
        return results

    if _is_sheet_dict(sheet):
        # Prefer typed_rows
        if sheet.get("typed_rows"):
            for r in sheet.get("typed_rows", []):
                if _matches_typed(r, match):
                    results.append(r)
            return results
        for r in sheet.get("rows", []):
            if _matches_raw(r, match):
                results.append(r)
    return results


def aggregate_sum(sheet, column: str, filters: dict[str, object] | None = None) -> float:
    """Sum numeric `column` across rows matching optional `filters`.

    Uses typed_rows when available for reliable numeric computation. If
    typed values are missing, attempts to coerce raw strings into numbers.
    """
    total = 0.0
    def _add_value(v):
        nonlocal total
        if v is None:
            return
        if isinstance(v, (int, float)):
            total += float(v)
            return
        # Try to coerce from string
        s = str(v).replace(",", "").strip()
        if not s:
            return
        try:
            total += float(s)
        except Exception:
            return

    if isinstance(sheet, ParsedTable):
        rows = sheet.rows
        if filters:
            rows = [r for r in rows if all(_normalize(str(r.get(k, ""))) == _normalize(str(v)) for k, v in filters.items())]
        for r in rows:
            _add_value(r.get(column))
        return total

    if _is_sheet_dict(sheet):
        if sheet.get("typed_rows"):
            rows = sheet.get("typed_rows", [])
            if filters:
                rows = [r for r in rows if all(
                    (str(r.get(k)) == str(v)) if isinstance(v, (int, float)) else (_normalize(str(r.get(k, ""))) == _normalize(str(v)))
                    for k, v in filters.items()
                )]
            for r in rows:
                _add_value(r.get(column))
            return total
        # Fall back to raw rows
        rows = sheet.get("rows", [])
        if filters:
            rows = [r for r in rows if all(_normalize(str(r.get(k, ""))) == _normalize(str(v)) for k, v in filters.items())]
        for r in rows:
            _add_value(r.get(column))
    return total


# --- Row selection & formatting helpers ---

def _get_rows_matching_filters(sheet, filters: dict[str, object] | None):
    """Return list of rows matching filters. Prefers typed_rows when available.
    Returned rows are the raw dicts (string-valued) for provenance display."""
    if not filters:
        # return all rows
        if _is_sheet_dict(sheet):
            return sheet.get("rows", [])
        elif isinstance(sheet, ParsedTable):
            return sheet.rows
        return []

    results = []
    if _is_sheet_dict(sheet) and sheet.get("typed_rows"):
        for tr, raw in zip(sheet.get("typed_rows", []), sheet.get("rows", [])):
            match = True
            for k, v in filters.items():
                rv = tr.get(k)
                if isinstance(rv, (int, float)) or isinstance(v, (int, float)):
                    try:
                        if float(rv) != float(v):
                            match = False
                            break
                    except Exception:
                        match = False
                        break
                else:
                    if _normalize(str(rv or "")) != _normalize(str(v or "")):
                        match = False
                        break
            if match:
                results.append(raw)
        return results
    # fallback to raw rows matching
    rows = sheet.get("rows", []) if _is_sheet_dict(sheet) else (sheet.rows if isinstance(sheet, ParsedTable) else [])
    for r in rows:
        match = True
        for k, v in filters.items():
            if _normalize(str(r.get(k, ""))) != _normalize(str(v or "")):
                match = False
                break
        if match:
            results.append(r)
    return results


def _format_number(value: float) -> str:
    """Format a numeric value with thousands separators. Keep integers without
    decimal places, otherwise show two decimal places."""
    try:
        if value is None:
            return ""
        if abs(value - int(value)) < 1e-9:
            return f"{int(value):,}"
        return f"{value:,.2f}"
    except Exception:
        return str(value)


# --- New: infer aggregate/filter from natural-language question ---
import typing

def _header_mentioned_in_question(header: str, ql: str) -> bool:
    """Return True if header (or an alias for it) is mentioned in the question text."""
    hnorm = _normalize(header)
    if hnorm in ql:
        return True
    # check aliases: if any alias maps into this header and the alias token is in question
    for alias, mapped in _HEADER_ALIASES.items():
        if mapped in hnorm and alias in ql:
            return True
    # check individual header words
    for w in re.findall(r"\w+", hnorm):
        if w in ql:
            return True
    return False


def infer_aggregate_spec(question: str, headers: list[str], sheet=None) -> tuple[dict[str, object] | None, str | None]:
    """Infer (filters, aggregate_column) from a natural-language question.

    Returns (filters_dict_or_None, agg_column_or_None). Conservative: only
    returns a spec when confident; otherwise returns (None, None).
    """
    ql = question.lower()
    # Find candidate aggregate column by explicit phrasing like 'total Profit' or 'sum of Profit'
    agg_col = None
    m_agg = re.search(r"(?:total|sum)(?: of)?\s+([a-z0-9 _-]+)", ql)
    if m_agg:
        target = m_agg.group(1).strip()
        # try resolving the target token to a known header using aliases
        resolved = resolve_header(target, headers)
        if resolved:
            agg_col = resolved
        else:
            # match against headers (prefer whole-word header matches)
            for h in headers:
                if _normalize(h) in _normalize(target) or _normalize(target) in _normalize(h):
                    agg_col = h
                    break
    # If still not found, prefer numeric-sounding headers by keyword
    if not agg_col:
        numeric_candidates = [h for h in headers if any(k in h.lower() for k in ["profit", "units", "sales", "gross", "amount", "price", "cogs"]) ]
        if numeric_candidates:
            agg_col = numeric_candidates[0]
    # final leave as None if nothing matched explicitly
    # agg_col may still be None and will be resolved later by calling code
    # Filters: attempt to extract one or more values (quoted or 'within' phrases)
    filters = {}
    # 1) quoted tokens: allow multiple quoted tokens
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
    values = []
    if quoted:
        values = [v.strip() for v in quoted if v.strip()]
    else:
        # 2) 'within the X records' or 'within X records' capturing multiple values
        m2 = re.search(r"within the ([^,\.]+?) records", ql)
        if not m2:
            m2 = re.search(r"within ([^,\.]+?) records", ql)
        if m2:
            text = m2.group(1).strip()
            # If the phrase ends with a header keyword (e.g., 'France country'),
            # treat the preceding text as the value and ignore the trailing header
            words = text.split()
            last = words[-1].lower()
            header_tokens = {h.lower() for h in headers} | set(_HEADER_ALIASES.keys())
            if len(words) > 1 and last in header_tokens:
                values = [" ".join(words[:-1]).strip()]
            else:
                # split on ' and ' or commas
                parts = re.split(r"\s+and\s+|\s*,\s*", text)
                values = [p.strip() for p in parts if p.strip()]
        else:
            # 3) simple 'within X and Y' or 'for X and Y' patterns
            m3 = re.search(r"(?:within|for|where)\s+([A-Za-z0-9 ,]+(?:\s+and\s+[A-Za-z0-9 ,]+)*)", ql)
            if m3:
                text = m3.group(1).strip()
                parts = re.split(r"\s+and\s+|\s*,\s*", text)
                values = [p.strip() for p in parts if p.strip()]

    # Map extracted values to headers conservatively
    def map_values_to_headers(values_list):
        mapped = {}
        if not values_list:
            return mapped
        # If sheet provided, prefer matching distinct_values
        if sheet is not None:
            for val in values_list:
                val_n = _normalize(val)
                candidates = []
                for h in headers:
                    vals = distinct_values(sheet, h)
                    for v in vals:
                        if _normalize(str(v)) == val_n:
                            candidates.append(h)
                            break
                if len(candidates) == 1:
                    mapped[candidates[0]] = val
                elif len(candidates) > 1:
                    # Ambiguous: prefer non-numeric headers
                    for c in candidates:
                        if not any(k in c.lower() for k in ["profit", "units", "sales", "gross", "amount", "price", "cogs"]):
                            mapped[c] = val
                            break
                    if val not in mapped.values():
                        mapped[candidates[0]] = val
        # If not yet mapped, try to assign by adjacent header mention in question
        for val in values_list:
            if val in mapped.values():
                continue
            val_n = _normalize(val)
            assigned = False
            for h in headers:
                # look for patterns like 'VALUE <header_word>' or '<header_word> VALUE' in the question
                h_words = " ".join(re.findall(r"\w+", _normalize(h)))
                if re.search(re.escape(val_n) + r"\s+" + re.escape(h_words), ql) or re.search(re.escape(h_words) + r"\s+" + re.escape(val_n), ql):
                    mapped[h] = val
                    assigned = True
                    break
            if assigned:
                continue
        # If still unmapped but number of mentioned headers equals values, map by order of appearance
        if len(mapped) < len(values_list):
            mentioned_headers = [h for h in headers if _header_mentioned_in_question(h, ql)]
            if len(mentioned_headers) == len(values_list):
                for h, v in zip(mentioned_headers, values_list):
                    if h not in mapped:
                        mapped[h] = v
        # Final fallback: assign to first non-numeric headers in order
        if len(mapped) < len(values_list):
            non_numeric = [h for h in headers if not any(k in h.lower() for k in ["profit", "units", "sales", "gross", "amount", "price", "cogs"]) ]
            i = 0
            for v in values_list:
                if v in mapped.values():
                    continue
                if i < len(non_numeric):
                    mapped[non_numeric[i]] = v
                    i += 1
                else:
                    # assign to any remaining header
                    for h in headers:
                        if h not in mapped:
                            mapped[h] = v
                            break
        return mapped

    if values:
        filters = map_values_to_headers(values)
    else:
        filters = {}
        m = re.search(r"['\"]([^'\"]+)['\"]", question)
        if m:
            val = m.group(1).strip()
            # Find header whose distinct values likely contain val (best-effort)
            for h in headers:
                # match header token in question (e.g., 'country records') or find val in header name
                if _header_mentioned_in_question(h, ql):
                    filters[h] = val
                    break
            if not filters:
                # try alias-driven detection: if question contains an alias, map it
                for alias in _HEADER_ALIASES:
                    if alias in ql:
                        resolved = resolve_header(alias, headers)
                        if resolved:
                            filters[resolved] = val
                            break
            if not filters:
                # as last resort, assign to first non-numeric header
                for h in headers:
                    if h.lower() not in (agg_col or "") and not any(k in h.lower() for k in ["profit", "units", "sales", "gross", "amount", "price", "cogs"]):
                        filters[h] = val
                        break
    if not filters:
        filters = None
    if not agg_col:
        agg_col = None
    return filters, agg_col



def compute_aggregate_for_question(sheet, question: str) -> tuple[str | None, dict | None]:
    """Try to compute an aggregate answer deterministically.

    Returns (answer_string_or_None, details_dict_or_None). details contains
    {'agg_column': ..., 'filters': {...}, 'value': float}
    """
    headers = get_headers_from_sheet(sheet)
    if not headers:
        return None, None

    filters, agg_col = infer_aggregate_spec(question, headers, sheet)
    # If infer couldn't decide agg_col but question names a header, try that
    if not agg_col:
        for h in headers:
            if h.lower() in question.lower():
                # prefer numeric headers
                if any(k in h.lower() for k in ["profit", "units", "sales", "gross", "amount", "price", "cogs"]):
                    agg_col = h
                    break
    if not agg_col:
        # final fallback: choose most numeric header via find_numeric_column
        hint = None
        for k in ["profit", "units", "sales", "gross", "amount", "price"]:
            if k in question.lower():
                hint = k
                break
        agg_col = find_numeric_column(ParsedTable(headers=headers, rows=sheet.get("rows", [])), hint=hint)

    if not agg_col:
        return None, None

    value = aggregate_sum(sheet, agg_col, filters)
    # Prepare provenance rows and counts
    matching_rows = _get_rows_matching_filters(sheet, filters)
    row_count = len(matching_rows)
    provenance = matching_rows[:5]  # include first 5 rows for inspection

    # Format numeric value nicely
    formatted = _format_number(value)

    # Format answer naturally (use formatted number)
    if filters:
        # choose a readable filter string
        filt_text = ", ".join(f"{k}={v}" for k, v in filters.items())
        answer = f"The total {agg_col} for {filt_text} is {formatted}"
    else:
        answer = f"The total {agg_col} is {formatted}"
    details = {"agg_column": agg_col, "filters": filters, "value": value, "formatted_value": formatted, "row_count": row_count, "provenance_rows": provenance}
    return answer, details