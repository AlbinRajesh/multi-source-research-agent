"""
PDF text extraction parser.
Extracts clean, page-aware text from PDF documents using PyMuPDF (fitz).

Tables are detected and extracted in row-structured "col | col | col" form
(matching the format csv_parser.py already uses), instead of raw
get_text() dumping — raw extraction destroys row/column association for
tabular PDFs (numbers and labels get flattened into one text stream with
no delimiter tying a value to its row), which was causing downstream
retrieval and generation to misread or scramble table data. Pages with no
detected table still fall back to plain text extraction.
"""

import logging
from pathlib import Path
import pdfplumber

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Reuse the chunker's conservative heading detection to build sections
# so section-level metadata aligns with how chunks are later tagged.
from src.ingestion.chunker import _split_into_sections


class PDFParseError(Exception):
    """Raised when a PDF cannot be parsed."""
    pass

def _extract_table_text_pdfplumber_fallback(pdfplumber_doc, page_num: int) -> tuple[list, list, list]:
    try:
        page = pdfplumber_doc.pages[page_num - 1]
        tables = page.find_tables(table_settings={
            "vertical_strategy": "lines_strict",
            "horizontal_strategy": "lines_strict",
            "explicit_vertical_lines": [],
            "explicit_horizontal_lines": [],
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
            "text_tolerance": 3,
            "text_vert_tolerance": 3,
        })
        if not tables:
            tables = page.find_tables(table_settings={
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
            })
    except Exception as e:
        logger.warning(f"pdfplumber fallback failed on page {page_num}: {e}")
        return [], [], []

    table_blocks = []
    bboxes = []
    headers = []
    for table in tables:
        try:
            rows = table.extract()
        except Exception as e:
            logger.warning(f"pdfplumber fallback: failed to extract a table: {e}")
            continue
        clean_rows = [r for r in rows if any(c for c in r if c)]
        if not clean_rows:
            continue
        clean_rows = [[" ".join(str(c).split()) if c else "" for c in r] for r in clean_rows]
        lines = [" | ".join(r) for r in clean_rows]
        block = "\n".join(lines)
        table_blocks.append(block)
        bboxes.append(table.bbox)
        headers.append(lines[0])

    return table_blocks, bboxes, headers


def _extract_table_text(page) -> tuple[list, list, list]:
    """
    Returns:
        (table_blocks, covered_bboxes, headers) — table_blocks is a list of
        row-structured strings (one per detected table, NOT yet joined,
        so callers can merge continuation tables across pages); headers[i]
        is table_blocks[i]'s first row, used to detect continuation.
    """
    table_blocks = []
    covered_bboxes = []
    headers = []

    try:
        tables = page.find_tables()
    except Exception as e:
        logger.warning(f"Table detection failed on a page: {e}")
        return [], [], []

    for table in tables.tables:
        try:
            rows = table.extract()
        except Exception as e:
            logger.warning(f"Failed to extract a detected table: {e}")
            continue

        if not rows:
            continue

        clean_rows = []
        for row in rows:
            clean_row = [
                " ".join(str(cell).split()) if cell is not None else ""
                for cell in row
            ]
            clean_rows.append(clean_row)

        clean_rows = [r for r in clean_rows if any(c for c in r)]
        if not clean_rows:
            continue

        lines = [" | ".join(row) for row in clean_rows]
        table_blocks.append("\n".join(lines))
        covered_bboxes.append(table.bbox)
        headers.append(lines[0])

    return table_blocks, covered_bboxes, headers


def _extract_non_table_text(page, covered_bboxes: list) -> list[tuple[float, str]]:
    """Returns list of (y0, text) prose blocks, excluding table regions."""
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return [(0.0, page.get_text().strip())]

    def _overlaps_table(block_bbox) -> bool:
        bx0, by0, bx1, by1 = block_bbox
        for tx0, ty0, tx1, ty1 in covered_bboxes:
            if bx0 < tx1 and bx1 > tx0 and by0 < ty1 and by1 > ty0:
                return True
        return False

    out = []
    for b in blocks:
        bbox = b[:4]
        text = b[4].strip() if len(b) > 4 else ""
        if text and not _overlaps_table(bbox):
            out.append((bbox[1], text))
    return out

def _merge_heading_fragments(
    prose_items: list[tuple[float, str]],
    covered_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[tuple[float, str]]:
    """
    Merge short, unpunctuated blocks (likely stray headings) into the
    next block — UNLESS a table sits between them, in which case leave
    both blocks standalone (safer than dragging text across a table).
    """
    prose_items = sorted(prose_items, key=lambda t: t[0])
    covered_bboxes = covered_bboxes or []
 
    def _table_between(y_start: float, y_end: float) -> bool:
        for _, ty0, _, ty1 in covered_bboxes:
            # Any overlap between (y_start, y_end) and a table's y-range
            # counts as "a table sits between these two blocks."
            if ty0 < y_end and ty1 > y_start:
                return True
        return False
 
    merged: list[tuple[float, str]] = []
    i = 0
    while i < len(prose_items):
        y0, text = prose_items[i]
        is_short = len(text.split()) < 15
        lacks_terminal_punct = not text.strip().endswith((".", "?", "!"))
 
        if i + 1 < len(prose_items):
            next_y0, next_text = prose_items[i + 1]
        else:
            next_y0, next_text = None, None
 
        if (is_short and lacks_terminal_punct and next_y0 is not None
                and not _table_between(y0, next_y0)):
            prose_items[i + 1] = (next_y0, f"{text}\n{next_text}")
            i += 1
            continue
 
        merged.append((y0, text))
        i += 1
 
    return merged
 

def parse_pdf(file_path: str) -> dict:
    """
    Extract all text content from a PDF file. Tables are extracted with
    row/column structure preserved; remaining prose is extracted as plain
    text, with an automated OCR fallback for image-heavy or unselectable pages.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A dictionary containing full_text, pages count, and structured sections.

    Raises:
        PDFParseError: If the file doesn't exist, isn't a valid PDF,
                       or contains no extractable text.
    """
    path = Path(file_path)

    if not path.exists():
        raise PDFParseError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise PDFParseError(f"Not a PDF file: {file_path}")

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise PDFParseError(f"Could not open PDF (corrupted or invalid): {e}")
    pdfplumber_doc = pdfplumber.open(file_path)
    if doc.page_count == 0:
        doc.close()
        raise PDFParseError(f"PDF has no pages: {file_path}")

    pages_text = []
    tables_found_total = 0
    prev_table_header = None
    prev_table_page_idx = None

    for page_num, page in enumerate(doc, start=1):
        try:
            table_blocks, covered_bboxes, headers = _extract_table_text(page)
            if not table_blocks:
                table_blocks, covered_bboxes, headers = _extract_table_text_pdfplumber_fallback(pdfplumber_doc, page_num)

            all_covered_bboxes = covered_bboxes

            if table_blocks and prev_table_header and headers[0] == prev_table_header:
                cont_rows = table_blocks[0].split("\n")[1:]
                if cont_rows:
                    pages_text[prev_table_page_idx] += "\n" + "\n".join(cont_rows)
                table_blocks = table_blocks[1:]
                headers = headers[1:]
                covered_bboxes = covered_bboxes[1:] if covered_bboxes else covered_bboxes

            prose_items = _extract_non_table_text(page, all_covered_bboxes)
            prose_items = _merge_heading_fragments(prose_items, all_covered_bboxes)
            table_items = [(bb[1], tb) for bb, tb in zip(covered_bboxes, table_blocks)]

            all_items = sorted(prose_items + table_items, key=lambda t: t[0])
            page_full = "\n\n".join(t[1] for t in all_items)

            # OCR Fallback if standard extraction yields virtually nothing on this page (likely scanned image)
            if len(page_full.strip()) < 30:
                try:
                    import pytesseract
                    from pdf2image import convert_from_path
                    images = convert_from_path(file_path, first_page=page_num, last_page=page_num)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0]).strip()
                        if ocr_text:
                            page_full = ocr_text
                            logger.info(f"Page {page_num}: Successfully recovered via OCR fallback.")
                except Exception as ocr_err:
                    logger.warning(f"OCR fallback failed on page {page_num}: {ocr_err}")

            if table_blocks:
                tables_found_total += 1

            if page_full:
                pages_text.append(page_full)
                if headers:
                    prev_table_header = headers[-1]
                    prev_table_page_idx = len(pages_text) - 1
                else:
                    prev_table_header = None
            else:
                logger.warning(f"Page {page_num} of {path.name} has no extractable text.")
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_num} of {path.name}: {e}")

    page_count = doc.page_count
    doc.close()
    pdfplumber_doc.close()

    full_text = "\n\n".join(pages_text)

    if not full_text.strip():
        raise PDFParseError(
            f"No extractable text found in {file_path}. "
            f"This may be a scanned/image-only PDF requiring OCR."
        )

    logger.info(f"Extracted {len(full_text)} characters from {page_count} page(s) in {path.name} "
                f"({tables_found_total} page(s) with detected table(s))")

    sections: list[dict] = []
    for page_num, page_full in enumerate(pages_text, start=1):
        page_sections = _split_into_sections(page_full)
        for sec_idx, (heading, body) in enumerate(page_sections):
            sections.append({
                "section_title": heading,
                "page_num": page_num,
                "section_index": sec_idx,
                "text": body,
            })

    return {
        "full_text": full_text,
        "pages": page_count,
        "sections": sections,
    }