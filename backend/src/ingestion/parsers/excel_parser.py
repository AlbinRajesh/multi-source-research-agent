"""
Excel spreadsheet text extraction parser.
Extracts table-aware text from .xlsx files using openpyxl, preserving
row/column structure so the LLM can understand tabular relationships.
"""

import logging
from pathlib import Path
import re

import openpyxl
from openpyxl.utils.exceptions import InvalidFileException

logger = logging.getLogger(__name__)


class ExcelParseError(Exception):
    """Raised when an Excel file cannot be parsed."""
    pass


def _clean(v) -> str:
    return str(v).replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()


def parse_excel(file_path: str) -> list[dict]:
    """
    Extract all text content from an .xlsx file, sheet by sheet,
    preserving row/column structure.

    For each non-empty sheet this now returns a dict containing:
      - sheet_name: original sheet name
      - text: rows joined by " | " (backwards compatible with chunker)
      - headers: list of header strings (first non-empty row by default)
      - rows: list of raw string-valued row dicts mapping header->cell_text
      - typed_rows: list of typed row dicts mapping header->python-typed value

    Args:
        file_path: Path to the .xlsx file.

    Returns:
        list[dict]: One entry per non-empty sheet with the keys described above.

    Raises:
        ExcelParseError: If the file doesn't exist, isn't a valid .xlsx,
                          or contains no data.
    """
    import datetime

    def _coerce_value(v):
        """Try to coerce common cell values to native Python types.
        Preserves numbers (int/float) and datetimes where possible; otherwise
        returns a cleaned string. """
        if v is None:
            return None
        # Already a native number or datetime from openpyxl
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, (datetime.date, datetime.datetime)):
            # Use ISO format for dates so downstream systems can parse easily
            return v.isoformat()
        # Otherwise treat as string and attempt numeric coercion
        s = _clean(v)
        # Accounting-style negatives: (1,234.56)
        if re.match(r"^\(.*\)$", s):
            inner = s[1:-1].replace(",", "").strip()
            if re.fullmatch(r"\d+(\.\d+)?", inner):
                return -float(inner) if ("." in inner) else -int(inner)
        # Strip common currency symbols and commas
        stripped = re.sub(r"[\$,£€]", "", s)
        stripped = stripped.replace(",", "")
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"-?\d+\.\d+", stripped):
            return float(stripped)
        return s

    path = Path(file_path)

    if not path.exists():
        raise ExcelParseError(f"File not found: {file_path}")

    if path.suffix.lower() not in (".xlsx", ".xlsm"):
        raise ExcelParseError(f"Not an .xlsx file: {file_path} "
                               f"(note: legacy .xls files are not supported)")

    try:
        workbook = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
    except InvalidFileException:
        raise ExcelParseError(f"Could not open Excel file (corrupted or invalid): {file_path}")
    except Exception as e:
        raise ExcelParseError(f"Unexpected error opening Excel file: {e}")

    sheets = []
    try:
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            row_values = []  # list of tuples as returned by openpyxl

            for row in sheet.iter_rows(values_only=True):
                if all(cell is None for cell in row):
                    continue
                row_values.append(row)

            if not row_values:
                logger.warning(f"Sheet '{sheet_name}' in {path.name} is empty.")
                continue

            # Build the human-readable text (backwards compatible)
            row_lines = [" | ".join(_clean(cell) if cell is not None else "" for cell in row) for row in row_values]
            text_block = "\n".join(row_lines)

            # Determine headers: prefer the first row if it contains at least two non-empty cells
            header_row = row_values[0]
            non_empty = sum(1 for c in header_row if c is not None and str(c).strip())
            if non_empty < 2 and len(row_values) > 1:
                header_row = row_values[1]
                data_rows = row_values[2:]
            else:
                data_rows = row_values[1:]

            headers = [(_clean(h) if h is not None else f"column_{i}") for i, h in enumerate(header_row)]

            rows = []
            typed_rows = []
            for r in data_rows:
                # Pad row to headers length
                cells = list(r) + [None] * max(0, len(headers) - len(r))
                raw_row = {}
                typed_row = {}
                for i, h in enumerate(headers):
                    raw_val = _clean(cells[i]) if cells[i] is not None else ""
                    raw_row[h] = raw_val
                    typed_row[h] = _coerce_value(cells[i])
                rows.append(raw_row)
                typed_rows.append(typed_row)

            sheets.append({
                "sheet_name": sheet_name,
                "text": text_block,
                "headers": headers,
                "rows": rows,
                "typed_rows": typed_rows,
            })
    finally:
        workbook.close()

    if not sheets:
        raise ExcelParseError(f"No extractable data found in {file_path}. All sheets appear empty.")

    logger.info(f"Extracted data from {len(sheets)} sheet(s) in {path.name}")
    return sheets