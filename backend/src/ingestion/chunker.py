"""
Text chunking for retrieval.
Splits extracted document text into overlapping chunks suitable for
embedding and retrieval, using token-based sizing (not raw character count)
for more consistent chunk sizes across different LLM tokenizers.

Table-formatted text (lines joined with " | ", as produced by the CSV and
PDF table parsers) is chunked line-by-line so a table row is never split
across two chunks — splitting mid-row was destroying the row's meaning
for retrieval and downstream reasoning.

Prose text is additionally split into heading-bounded sections before the
token sliding-window runs, so a chunk never silently spans two unrelated
topics — each chunk is tagged with its nearest heading (e.g.
"[Section: Feline Chronic Kidney Disease]") for retrieval-time context.
This is bounded, not a full structural rewrite: heading detection only
decides WHERE the window resets, never how big a chunk gets — the
sliding-window size cap still applies inside every section, so a missed
or malformed heading degrades to "no tag" rather than an oversized or
malformed chunk. Documents with no detectable headings behave exactly
as before.
"""

import logging
import re
from dataclasses import dataclass

import tiktoken

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 500      # tokens per chunk
DEFAULT_CHUNK_OVERLAP = 50    # tokens shared between consecutive chunks
DEFAULT_ROW_OVERLAP_LINES = 2  # lines carried over between table chunks
MIN_EFFECTIVE_CHUNK_SIZE = 100  # floor after reserving tokens for a heading tag

_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    chunk_index: int
    token_count: int


def _looks_tabular(text: str) -> bool:
    """Heuristic: text is treated as table data if most non-blank lines
    contain a ' | ' column separator (the format used by csv_parser.py
    and the table-extraction path in pdf_parser.py)."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    delimited = sum(1 for l in lines if " | " in l)
    return delimited / len(lines) > 0.6


def _find_header_line_index(lines: list[str]) -> int:
    """
    Find the first line that looks like a real table header — pipe-
    delimited with mostly non-empty cells — instead of blindly assuming
    lines[0] is the header. This matters when a document mixes prose
    (e.g. an "About" paragraph) above the actual table on the same page:
    lines[0] would be a sentence, not a header, causing every chunk to
    be stamped with the wrong "header" and breaking cross-chunk table
    stitching downstream. Falls back to 0 if nothing better is found,
    preserving old behavior for purely-tabular documents.
    """
    for i, line in enumerate(lines):
        if " | " not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        non_empty = sum(1 for c in cells if c)
        if non_empty >= 2 and non_empty / len(cells) > 0.6:
            return i
    return 0


# ---------------------------------------------------------------------------
# Heading detection for prose section splitting
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    re.compile(r'^#{1,6}\s+\S.*'),                                  # Markdown: # Heading
    re.compile(r'^(?:chapter|section|part)\s+[IVXLC\d]+[\.\):]?\s*(?:[A-Z][^.!?]{0,50})?$', re.IGNORECASE),  # "Chapter 3" / "Section 3: Overview" / "Part II — Definitions" — rejects sentence continuations like "Section 12(a) provides that..."
    re.compile(r'^page\s+\d+\s*:\s*\S.*', re.IGNORECASE),           # "Page 10: Conclusion"
    re.compile(r'^(?:HLD\s+)?System\s+\d+\s*:\s*\S.*$', re.IGNORECASE),  # "System 3: Instagram" / "HLD System 3: Instagram" — the named-system headings used in comparison-style HLD documents. Requires the literal word "System" + a number + colon, so it won't false-positive on generic numbered clauses the way a bare "3." pattern would.
]
# Numbered ("1.", "2)") and bare ALL-CAPS patterns were dropped: on
# enterprise documents (policies, contracts) numbered clauses and
# recurring page boilerplate ("CONFIDENTIAL", "DRAFT", "PAGE 3 OF 12")
# false-positive-matched these, silently fragmenting prose into fake
# "sections" at every clause or page break. The remaining patterns
# require unambiguous structural formatting, so misses just mean no
# tag (safe) instead of a wrong section boundary (actively misleading).


def _looks_like_heading(line: str) -> bool:
    """
    Heuristic-only, deliberately conservative — a false negative just
    means no section tag (identical to old behavior); a false positive
    would wrongly cut a section, so patterns are kept narrow and require
    fairly specific formatting rather than guessing.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False

    for pat in _HEADING_PATTERNS:
        if pat.match(stripped):
            return True

    return False

def _clean_heading_text(line: str) -> str:
    """Strip Markdown '#' markers and surrounding whitespace for display."""
    return line.strip().lstrip("#").strip()


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """
    Split prose text into (heading, section_body) pairs at detected
    heading lines. Content before the first heading (or all content, if
    no headings are found) gets heading=None — callers must treat that
    as "no tag" and fall back to unmodified chunking.
    """
    lines = text.split("\n")
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if _looks_like_heading(line):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = _clean_heading_text(line)
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return [(h, body) for h, body in sections if body.strip()]


def _chunk_plain_text(text: str, chunk_size: int, chunk_overlap: int, start_index: int = 0) -> list[Chunk]:
    """Token-based sliding-window chunking for ordinary prose text.
    Extracted out of chunk_text() so _chunk_tabular() can reuse it for
    prose sections (e.g. an intro paragraph) that appear before a table
    in the same document, instead of discarding that text."""
    tokens = _encoding.encode(text)
    total_tokens = len(tokens)

    chunks = []
    start = 0
    chunk_index = start_index

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_str = _encoding.decode(chunk_tokens).strip()

        if chunk_str:
            chunks.append(Chunk(
                text=chunk_str,
                chunk_index=chunk_index,
                token_count=len(chunk_tokens),
            ))
            chunk_index += 1

        if end == total_tokens:
            break
        start += chunk_size - chunk_overlap

    return chunks


def _chunk_plain_text_with_sections(text: str, chunk_size: int, chunk_overlap: int, start_index: int = 0) -> list[Chunk]:
    """
    Section-bounded wrapper around _chunk_plain_text. Splits text at
    detected headings first, then runs the normal sliding window inside
    each section (never across a section boundary), tagging every
    resulting chunk with "[Section: <heading>]" so the tag becomes part
    of what gets embedded — disambiguating chunks whose own text alone
    is ambiguous out of context.

    Bounded by design: the window's chunk_size cap still applies inside
    every section, so a long/undetected section still produces normally-
    sized chunks — it just won't carry a heading tag. No single failure
    mode (missed heading, false-positive heading) can produce an
    oversized or malformed chunk.
    """
    sections = _split_into_sections(text)

    # No headings detected at all -> identical to old behavior.
    if len(sections) <= 1 and sections and sections[0][0] is None:
        return _chunk_plain_text(text, chunk_size, chunk_overlap, start_index)

    all_chunks: list[Chunk] = []
    chunk_index = start_index

    for heading, section_text in sections:
        if not section_text:
            continue

        prefix = f"[Section: {heading}]\n" if heading else ""
        prefix_tokens = len(_encoding.encode(prefix)) if prefix else 0
        effective_chunk_size = max(chunk_size - prefix_tokens, MIN_EFFECTIVE_CHUNK_SIZE)
        effective_overlap = min(chunk_overlap, effective_chunk_size - 1)

        sub_chunks = _chunk_plain_text(section_text, effective_chunk_size, effective_overlap, 0)

        for c in sub_chunks:
            if prefix:
                c.text = f"{prefix}{c.text}"
                c.token_count += prefix_tokens
            c.chunk_index = chunk_index
            chunk_index += 1
            all_chunks.append(c)

    return all_chunks


def _chunk_tabular(text: str, chunk_size: int, overlap_lines: int) -> list[Chunk]:
    """
    Line-aware chunking — never splits a row, keeps the header line
    repeated at the top of every chunk so each chunk is self-describing.
    Reverted from per-block splitting: multi-page/multi-table reconciliation
    is now handled at query time by table_utils.parse_and_merge_tables(),
    via normalized header stitching — not at chunk time. Splitting chunks
    per \n\n-block caused rows to be silently dropped whenever a page's
    table lacked byte-exact header text, which regressed PDF aggregate
    accuracy significantly.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return []

    header_idx = _find_header_line_index(lines)
    header = lines[header_idx]
    body_lines = lines[header_idx + 1:]

    # Lines before the real header (e.g. an intro/"About" paragraph that
    # shares a page with the table) aren't table rows — chunk them as
    # ordinary prose instead of silently dropping them, so prose_lookup
    # questions about that content still have something to retrieve.
    prose_chunks: list[Chunk] = []
    if header_idx > 0:
        prose_text = "\n".join(lines[:header_idx]).strip()
        if prose_text:
            prose_chunks = _chunk_plain_text(prose_text, chunk_size, DEFAULT_CHUNK_OVERLAP)

    chunks = []
    chunk_index = 0
    i = 0
    while i < len(body_lines):
        current_lines = [header]
        current_tokens = len(_encoding.encode(header))
        start_i = i

        while i < len(body_lines):
            line_tokens = len(_encoding.encode(body_lines[i]))
            if current_tokens + line_tokens > chunk_size and len(current_lines) > 1:
                break
            current_lines.append(body_lines[i])
            current_tokens += line_tokens
            i += 1

        chunk_str = "\n".join(current_lines).strip()
        chunks.append(Chunk(text=chunk_str, chunk_index=chunk_index, token_count=current_tokens))
        chunk_index += 1

        if i >= len(body_lines):
            break
        i = max(i - overlap_lines, start_i + 1)

    # Prepend prose chunks and renumber chunk_index so it's contiguous
    # across both sections.
    all_chunks = prose_chunks + chunks
    for idx, c in enumerate(all_chunks):
        c.chunk_index = idx

    return all_chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split text into overlapping, token-sized chunks. Documents are split
    into blank-line-separated blocks first, and EACH BLOCK is classified
    independently as tabular or prose — a document that mixes prose and
    tables (the common case for docx/pdf) must not have its tables
    silently discarded because prose dominates the overall line count.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text or not text.strip():
        logger.warning("Received empty text to chunk; returning no chunks.")
        return []

    blocks = [b for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return []

    all_chunks: list[Chunk] = []
    for block in blocks:
        if _looks_tabular(block):
            block_chunks = _chunk_tabular(block, chunk_size, DEFAULT_ROW_OVERLAP_LINES)
        else:
            block_chunks = _chunk_plain_text_with_sections(block, chunk_size, chunk_overlap)
        all_chunks.extend(block_chunks)

    for idx, c in enumerate(all_chunks):
        c.chunk_index = idx

    logger.info(f"Split into {len(all_chunks)} chunk(s) across {len(blocks)} block(s) "
                f"(size={chunk_size}, overlap={chunk_overlap})")
    return all_chunks