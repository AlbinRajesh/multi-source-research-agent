"""
CSV text extraction parser.
Extracts table-aware text from .csv files using pandas, preserving
row/column structure so the LLM can understand tabular relationships.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class CSVParseError(Exception):
    """Raised when a CSV file cannot be parsed."""
    pass


def parse_csv(file_path: str) -> str:
    """
    Extract all data from a .csv file, preserving column headers and
    row structure.

    Args:
        file_path: Path to the .csv file.

    Returns:
        Extracted text, with the header row and each data row joined
        by " | " to preserve table structure.

    Raises:
        CSVParseError: If the file doesn't exist, isn't valid CSV,
                        or contains no data.
    """
    path = Path(file_path)

    if not path.exists():
        raise CSVParseError(f"File not found: {file_path}")

    if path.suffix.lower() != ".csv":
        raise CSVParseError(f"Not a .csv file: {file_path}")

    try:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise CSVParseError(f"CSV file is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise CSVParseError(f"Could not parse CSV (malformed structure): {e}")
    except UnicodeDecodeError:
        # Common with CSVs exported from older Windows tools (Excel, etc.)
        try:
            df = pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding="latin-1")
            logger.warning(f"{path.name} was not UTF-8 encoded; fell back to latin-1.")
        except Exception as e:
            raise CSVParseError(f"Could not decode CSV file (unsupported encoding): {e}")
    except Exception as e:
        raise CSVParseError(f"Unexpected error reading CSV: {e}")

    if df.empty:
        raise CSVParseError(f"No data rows found in {file_path}.")

    def _clean(v: str) -> str:
        return str(v).replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()

    lines = [" | ".join(_clean(col) for col in df.columns)]  # header row
    for _, row in df.iterrows():
        lines.append(" | ".join(_clean(val) for val in row))

    full_text = "\n".join(lines)

    logger.info(f"Extracted {len(df)} row(s), {len(df.columns)} column(s) from {path.name} "
                f"({len(full_text)} characters)")
    return full_text