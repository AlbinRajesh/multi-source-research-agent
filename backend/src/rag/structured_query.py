"""
Structured aggregate computation engine — DuckDB over a cleaned pandas
DataFrame built from a ParsedTable.

Why this exists (replaces the old sum_column()/find_numeric_column()
hand-written arithmetic in table_utils.py):
  - DuckDB does correct sum/avg/min/max/count for free — we stop
    hand-writing each aggregation and stop being one bug away from a
    wrong number.
  - It runs in milliseconds against an in-memory DataFrame. No server,
    no extra infra, fully local — fits the existing Qdrant/Ollama stack.
  - It supports WHOLE-TABLE aggregates (no entity filter) and grouped
    comparisons ("which country has higher profit") in one engine,
    which the old code could not do at all.
  - The LLM never touches the actual numbers. It only helps pick which
    column/entity the question is about; the arithmetic is 100%
    deterministic code.

Contract with callers (agent/nodes.py):
    run_aggregate(...)   -> (answer_text, evidence_dict) | None
    run_comparison(...)  -> (answer_text, evidence_dict) | None
`None` means "could not compute this deterministically" — the caller
should fall back to the existing LLM path, same as the old code did.
"""

import logging
import re

import duckdb
import pandas as pd

from retrieval.table_utils import ParsedTable, _normalize

logger = logging.getLogger(__name__)

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


# ---------------------------------------------------------------------------
# Step 1: turn table_utils' string-only rows into a typed DataFrame
# ---------------------------------------------------------------------------

def format_number(value: float, column_name: str = "") -> str:
    col_lower = column_name.lower()
    if any(k in col_lower for k in ["price", "amount", "revenue", "cost", "profit", "sales"]):
        return f"${value:,.2f}"
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"

def _clean_numeric_cell(v: str) -> float | None:
    """Best-effort numeric parse: strips commas/currency/percent signs,
    and converts accounting-style '(123.45)' into -123.45. Returns None
    for anything that isn't actually a number, e.g. text cells."""
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    v = v.replace(",", "").replace("$", "").replace("%", "").strip()
    if v.startswith("(") and v.endswith(")"):
        v = "-" + v[1:-1]
    return float(v) if _NUMERIC_RE.fullmatch(v) else None


def table_to_dataframe(table: ParsedTable) -> pd.DataFrame:
    """
    Builds a DataFrame from ParsedTable rows. Every column stays as raw
    text (for filtering/grouping), and any column that is mostly numeric
    also gets a parallel "<col>__num" float column DuckDB can aggregate
    on. A column is only treated as numeric if >60% of its non-blank
    cells actually parse as a number — avoids miscasting a text column
    that happens to contain a stray digit.
    """
    headers = _dedupe_headers(table.headers)
    df = pd.DataFrame(table.rows, columns=table.headers)
    df.columns = headers   # apply deduped names

    for col in headers:
        col_data = df[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        parsed = col_data.map(_clean_numeric_cell)
        non_blank = col_data.astype(str).str.strip() != ""
        if non_blank.sum() == 0:
            continue
        ratio = parsed[non_blank].notna().sum() / non_blank.sum()
        if ratio > 0.6:
            df[col + "__num"] = parsed

    return df

def _dedupe_headers(headers: list[str]) -> list[str]:
    seen = {}
    result = []
    for h in headers:
        if h not in seen:
            seen[h] = 0
            result.append(h)
        else:
            seen[h] += 1
            result.append(f"{h}_{seen[h]}")
    return result


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Original header names (without the __num suffix) that have a
    computable numeric counterpart."""
    return [c[: -len("__num")] for c in df.columns if c.endswith("__num")]


# ---------------------------------------------------------------------------
# Step 2: figure out which column and which rows the question means
# ---------------------------------------------------------------------------

def _singularize(word: str) -> str:
    """Crude plural stripper so 'segments' in a question matches a
    'Segment' header — good enough for English column names without
    pulling in a full NLP stemmer."""
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def _word_overlap_score(a_words: set[str], b_words: set[str]) -> int:
    a_stems = {_singularize(w) for w in a_words}
    b_stems = {_singularize(w) for w in b_words}
    return len(a_stems & b_stems)


def _best_overlap_column(columns: list[str], query: str, hint: str | None = None) -> str | None:
    """Shared column-picking logic: prefer an explicit hint substring
    match, else the column whose header words overlap most with the
    question (plural-tolerant), else None (caller decides the default)."""
    if not columns:
        return None
    if hint:
        for c in columns:
            if hint.lower() in c.lower():
                return c
    query_words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
    best, best_score = None, 0
    for c in columns:
        header_words = set(re.findall(r"\w+", c.lower()))
        score = _word_overlap_score(query_words, header_words)
        if score > best_score:
            best, best_score = c, score
    return best
def _pick_group_column(df: pd.DataFrame, text_cols: list[str], query: str, hint: str | None = None) -> str | None:
    """Group-by column selection for comparisons. Header-word overlap
    alone misses cases like 'profit in 2014 vs 2013' -> 'Year' column,
    since '2014' doesn't appear in the header text at all. This also
    checks whether query tokens appear as actual CELL VALUES in each
    candidate column, and downranks columns that are near-unique per row
    (id-like columns aren't valid group-by targets — grouping by a
    column where every row is its own group is never what a comparison
    question means)."""
    if not text_cols:
        return None

    header_pick = _best_overlap_column(text_cols, query, hint)

    query_words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 1}
    n_rows = max(len(df), 1)

    best_col, best_score = None, -1
    for c in text_cols:
        values = df[c].astype(str).str.strip().str.lower()
        distinct = values.nunique()
        cardinality_ratio = distinct / n_rows

        # Near-unique columns (e.g. IDs, serials) are never valid
        # group-by targets for a comparison question.
        if cardinality_ratio > 0.5:
            continue

        value_hits = sum(1 for v in values.unique() if v in query_words)
        score = value_hits * 10  # value match is a much stronger signal than header overlap
        if c == header_pick:
            score += 1

        if score > best_score:
            best_col, best_score = c, score

    if best_col is not None and best_score > 0:
        return best_col

    # No value-based signal at all -> fall back to header overlap, but
    # never return a near-unique column even as a last resort.
    if header_pick:
        distinct = df[header_pick].astype(str).nunique()
        if distinct / n_rows <= 0.5:
            return header_pick

    # Final fallback: first genuinely categorical (low-cardinality) column.
    for c in text_cols:
        distinct = df[c].astype(str).nunique()
        if distinct / n_rows <= 0.5:
            return c
    return text_cols[0]


def guess_target_column(df: pd.DataFrame, query: str, hint: str | None = None) -> str | None:
    """Picks the numeric column the question is actually asking about.
    Falls back to the first available numeric column if nothing matches."""
    candidates = numeric_columns(df)
    if not candidates:
        return None
    return _best_overlap_column(candidates, query, hint) or candidates[0]


def guess_target_column_categorical(df: pd.DataFrame, query: str, hint: str | None = None) -> str | None:
    """For count_distinct: pick a text (non-numeric) column, biased by
    query/header word overlap (plural-tolerant), falling back to the
    last text column (category/label columns are conventionally
    rightmost in these tables, same heuristic the old code used)."""
    text_cols = [c for c in df.columns if not c.endswith("__num")]
    if not text_cols:
        return None
    return _best_overlap_column(text_cols, query, hint) or text_cols[-1]


def filter_by_entity(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Row filter matching table_utils.find_matching_rows semantics —
    exact cell match preferred, substring as fallback — vectorized over
    the DataFrame instead of a Python loop."""
    entity_norm = _normalize(entity)
    if not entity_norm:
        return df.iloc[0:0]

    text_cols = [c for c in df.columns if not c.endswith("__num")]
    if not text_cols:
        return df.iloc[0:0]
    norm_df = df[text_cols].astype(str).apply(lambda col: col.map(_normalize))

    exact_mask = (norm_df == entity_norm).any(axis=1)
    if exact_mask.any():
        return df[exact_mask]

    def _contains(col: pd.Series) -> pd.Series:
        return col.apply(lambda v: bool(v) and (entity_norm in v or v in entity_norm))

    sub_mask = norm_df.apply(_contains).any(axis=1)
    return df[sub_mask]


# ---------------------------------------------------------------------------
# Step 3: run the actual math in DuckDB
# ---------------------------------------------------------------------------

def run_aggregate(
    table: ParsedTable,
    mode: str,  # "sum" | "average" | "min" | "max" | "count" | "count_distinct"
    query: str,
    entity: str | None = None,
    column_hint: str | None = None,
) -> tuple[str, dict] | None:
    """
    Computes the requested aggregate with DuckDB over a cleaned
    DataFrame. Works with OR without an entity filter — a whole-table
    aggregate (no entity given) is now a first-class case, not a
    fallback failure like in the old code.

    Returns (answer_text, evidence_dict), or None if it can't be
    computed deterministically (caller falls back to the LLM path).
    """
    df = table_to_dataframe(table)
    if df.empty:
        return None

    scope_label = "the entire dataset"
    if entity:
        filtered = filter_by_entity(df, entity)
        if filtered.empty:
            return None
        df = filtered
        scope_label = entity

    if mode == "count":
        n = len(df)
        answer = f"There are {n} matching row(s) for {scope_label} in the document."
        return answer, {"row_count": n, "scope": scope_label}

    if mode == "count_distinct":
        col = guess_target_column_categorical(df, query, column_hint)
        if col is None:
            return None
        distinct_vals = sorted(v for v in df[col].astype(str).str.strip().unique() if v)
        if not distinct_vals:
            return None
        answer = (
            f"There are {len(distinct_vals)} distinct value(s) in the "
            f"'{col}' column: {', '.join(distinct_vals)}."
        )
        return answer, {"column": col, "values": distinct_vals}

    # sum / average / min / max all need a numeric target column
    col = guess_target_column(df, query, column_hint)
    if col is None:
        return None
    num_col = col + "__num"

    con = duckdb.connect()
    con.register("t", df)
    sql_fn = {"sum": "SUM", "average": "AVG", "min": "MIN", "max": "MAX"}[mode]
    result = con.execute(f'SELECT {sql_fn}("{num_col}") FROM t').fetchone()[0]
    con.close()

    if result is None:
        return None

    verb = {"sum": "total", "average": "average", "min": "minimum", "max": "maximum"}[mode]
    value = round(float(result), 2)
    answer = f"The {verb} {col} for {scope_label} is {format_number(value, col)}."
    return answer, {
        "column": col,
        "value": value,
        "scope": scope_label,
        "rows_considered": len(df),
    }
def run_range(table: ParsedTable, query: str, entity: str | None = None,
              column_hint: str | None = None) -> tuple[str, dict] | None:
    """Computes max-min ('range'/'spread') for one column — distinct from
    run_comparison, which compares totals across two different groups."""
    df = table_to_dataframe(table)
    if df.empty:
        return None

    scope_label = "the entire dataset"
    if entity:
        filtered = filter_by_entity(df, entity)
        if filtered.empty:
            return None
        df = filtered
        scope_label = entity

    col = guess_target_column(df, query, column_hint)
    if col is None:
        return None
    num_col = col + "__num"

    con = duckdb.connect()
    con.register("t", df)
    row = con.execute(f'SELECT MIN("{num_col}"), MAX("{num_col}") FROM t').fetchone()
    con.close()

    if row is None or row[0] is None:
        return None
    lo, hi = float(row[0]), float(row[1])
    answer = f"The range of {col} for {scope_label} is {format_number(hi - lo, col)} (max {format_number(hi, col)} minus min {format_number(lo, col)})."
    return answer, {"column": col, "min": lo, "max": hi, "range": hi - lo, "scope": scope_label}

_COMPARE_ENTITY_RE = re.compile(
    r'\b([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*)\b'
)

def _extract_named_entities_for_compare(query: str, df: pd.DataFrame, text_cols: list[str]) -> list[str]:
    """For 'X vs Y' / 'X and Y' comparison questions, find which specific
    cell values from the table are actually named in the query — so the
    comparison is scoped to those two (or more) entities instead of
    silently ranking the whole table and returning the global top-2,
    which answers a different question than the one asked."""
    if not text_cols:
        return []

    all_values = set()
    for c in text_cols:
        all_values.update(v for v in df[c].astype(str).str.strip().unique() if v)

    query_lower = query.lower()
    matched = [v for v in all_values if v and v.lower() in query_lower]
    # Prefer longer matches first (e.g. "United States of America" over
    # a coincidental short substring) and cap at a sane number.
    matched.sort(key=len, reverse=True)
    return matched[:6]

def run_comparison(
    table: ParsedTable,
    query: str,
    group_col_hint: str | None = None,
    value_col_hint: str | None = None,
) -> tuple[str, dict] | None:
    """
    Handles "which X has higher/lower Y" questions — groups by a
    category column and computes each group's total, instead of
    forcing the LLM to eyeball two sums and compare them itself
    (this was the cause of the distractor_disambiguation failures).
    """
    df = table_to_dataframe(table)
    if df.empty:
        return None

    text_cols = [c for c in df.columns if not c.endswith("__num")]
    group_col = _pick_group_column(df, text_cols, query, group_col_hint)
    value_col = guess_target_column(df, query, value_col_hint)
    if group_col is None or value_col is None:
        return None
    num_col = value_col + "__num"

    named_entities = _extract_named_entities_for_compare(query, df, text_cols)
    con = duckdb.connect()
    con.register("t", df)
    if named_entities:
        placeholders = ", ".join(f"'{e}'" for e in named_entities)
        result = con.execute(
            f'SELECT "{group_col}", SUM("{num_col}") AS total FROM t '
            f'WHERE "{group_col}" IN ({placeholders}) '
            f'GROUP BY "{group_col}" ORDER BY total DESC'
        ).fetchall()
    else:
        result = con.execute(
            f'SELECT "{group_col}", SUM("{num_col}") AS total FROM t '
            f'GROUP BY "{group_col}" ORDER BY total DESC'
        ).fetchall()
    con.close()

    if not result:
        return None

    top_name, top_val = result[0]
    answer = f"{top_name} has the highest total {value_col} at {format_number(top_val, value_col)}."
    if len(result) > 1:
        second_name, second_val = result[1]
        diff = top_val - second_val
        answer += f" {second_name} is next at {format_number(second_val, value_col)}, a difference of {format_number(diff, value_col)}."

    return answer, {
        "group_column": group_col,
        "value_column": value_col,
        "ranking": result[:5],
    }
def _extract_column_and_value_for_exists(
    query: str, df: pd.DataFrame, text_cols: list[str]
) -> tuple[str | None, str | None]:
    """For 'does column X contain value Y' questions, find which real
    column and which candidate value the query is actually naming —
    same overlap-matching approach as _pick_group_column, but looking
    for a specific VALUE match rather than a group-by target."""
    if not text_cols:
        return None, None

    query_lower = query.lower()

    # Find the column: prefer a header whose name is literally quoted
    # or named in the query ("Discount Band column", "Segment column").
    col_pick = _best_overlap_column(text_cols, query)

    # Find the candidate value: look for quoted phrases first (most
    # reliable signal — "'Very High' category", "'Iris-hybrida'"),
    # then fall back to checking every cell value across all text
    # columns for a literal substring match in the query.
    quoted = re.findall(r"['\"]([^'\"]{2,60})['\"]", query)
    if quoted:
        return col_pick, quoted[0].strip()

    # No quotes — scan all cell values across text columns for the
    # longest one that appears literally in the query. This catches
    # unquoted phrasing like "does it include millimeters or inches".
    all_values = set()
    for c in text_cols:
        all_values.update(v for v in df[c].astype(str).str.strip().unique() if v)
    matched = [v for v in all_values if v and v.lower() in query_lower]
    matched.sort(key=len, reverse=True)
    if matched:
        return col_pick, matched[0]

    return None, None


def run_exists_check(table: ParsedTable, query: str) -> tuple[str, dict] | None:
    """Deterministic existence/negation check — 'does column X contain
    value Y' / 'is there a Z' answered via exact table scan instead of
    asking the LLM to read every row itself, which small local models
    do unreliably even with full context (they tend to give up and say
    'not found' instead of confirming absence with real detail).

    Returns (answer_text, evidence_dict), or None if it can't be
    resolved deterministically (caller falls back to the LLM path).
    """
    df = table_to_dataframe(table)
    if df.empty:
        return None

    text_cols = [c for c in df.columns if not c.endswith("__num")]
    target_col, target_value = _extract_column_and_value_for_exists(query, df, text_cols)
    if target_col is None or target_value is None:
        return None

    all_values = sorted(v for v in df[target_col].astype(str).str.strip().unique() if v)
    if not all_values:
        return None

    target_norm = _normalize(target_value)
    normalized_values = {_normalize(v) for v in all_values}
    found = target_norm in normalized_values

    if found:
        answer = f"Yes — '{target_value}' is present in the '{target_col}' column."
    else:
        answer = f"No — the '{target_col}' column only contains: {', '.join(all_values)}."

    return answer, {
        "column": target_col,
        "checked_value": target_value,
        "found": found,
        "all_values": all_values,
    }
def run_first_occurrence(
    table: ParsedTable,
    query: str,
    id_col_hint: str | None = None,
    order: str = "first",  # "first" | "last"
) -> tuple[str, dict] | None:
    """
    Handles 'at which <id_col> does <column> first/last become/change to
    <value>' and transition questions ('first change from A to B').
    Resolved by exact ordered scan in DuckDB — never asked of the LLM,
    since small local models unreliably hold state while scanning many
    rows (the csv1_07 failure class: correct row was in context but the
    LLM picked a structurally-similar-but-wrong later occurrence).
    """
    df = table_to_dataframe(table)
    if df.empty:
        return None

    text_cols = [c for c in df.columns if not c.endswith("__num")]
    if not text_cols:
        return None

    id_col = id_col_hint
    if id_col is None:
        for c in df.columns:
            if c.lower() in ("id", "row", "index", "no", "no.", "#"):
                id_col = c
                break
    if id_col is None:
        id_col = df.columns[0]

    # Try to detect a transition pattern: "from A to B" / "changes to B"
    transition_match = re.search(
        r'from\s+([\w\-]+)\s+to\s+([\w\-]+)', query, re.IGNORECASE
    )
    quoted = re.findall(r"['\"]([^'\"]{2,60})['\"]", query)

    target_col = _best_overlap_column(text_cols, query)
    if target_col is None:
        return None

    sort_key = id_col + "__num" if (id_col + "__num") in df.columns else id_col
    df_sorted = df.sort_values(by=sort_key).reset_index(drop=True)
    values = df_sorted[target_col].astype(str).str.strip().tolist()
    ids = df_sorted[id_col].tolist()

    if transition_match:
        from_val, to_val = transition_match.group(1), transition_match.group(2)
        rng = range(len(values) - 1) if order == "first" else range(len(values) - 2, -1, -1)
        for i in rng:
            if _normalize(values[i]) == _normalize(from_val) and _normalize(values[i + 1]) == _normalize(to_val):
                answer = f"The {target_col} column first changes from {from_val} to {to_val} at {id_col} {ids[i+1]}."
                return answer, {"id_column": id_col, "id_value": ids[i + 1], "column": target_col}
        return None
    # Otherwise: "first/last row where column == value"
    target_value = quoted[0] if quoted else None
    if target_value is None:
        return None
    rng = range(len(values)) if order == "first" else range(len(values) - 1, -1, -1)
    for i in rng:
        if _normalize(values[i]) == _normalize(target_value):
            answer = f"The {order} row where {target_col} is '{target_value}' is at {id_col} {ids[i]}."
            return answer, {"id_column": id_col, "id_value": ids[i], "column": target_col}
    return None