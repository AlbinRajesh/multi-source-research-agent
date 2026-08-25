"""
Image text extraction parser (OCR).
Extracts text from image files using Tesseract OCR — phase 1 scope:
clean printed text only. Engineering drawings / complex diagrams are
explicitly out of scope for this phase (see project notes).
"""

import logging
from pathlib import Path
import re

import pytesseract
from pytesseract import Output
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError

# Explicitly point to the Tesseract executable (PATH setup was unreliable on this machine)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# Minimum characters to consider OCR output "real" text rather than noise.
MIN_TEXT_LENGTH = 3

# Primary and fallback page-segmentation modes.
# 6 = "assume a single uniform block of text" — good default for
# documents/screenshots/tables.
# 4 = "assume a single column of variable-size text" — sometimes wins
# on table-like layouts where 6 merges columns incorrectly.
_OCR_CONFIG_PRIMARY = "--psm 6"
_OCR_CONFIG_FALLBACK = "--psm 4"

# If mean word confidence from the primary pass is below this, try the
# fallback config and keep whichever scored higher.
_CONFIDENCE_RETRY_THRESHOLD = 65.0

# Below this width, upscale before OCR. Small/compressed images lose
# enough detail that Tesseract's character recognition degrades sharply,
# especially for digits and punctuation like '|'.
_MIN_OCR_WIDTH = 1800
OCR_DEBUG_LOG: list[dict] = []

# --- Two-column layout detection ---------------------------------------
# Neither --psm 6 nor --psm 4 actually detects and separates side-by-side
# columns (e.g. "Seller" info next to "Client" info on an invoice) — both
# read left-to-right across the full image width, which interleaves the
# two columns' text line by line instead of reading one column fully
# before the other. These constants tune a lightweight gutter-detection
# pass that catches this case and OCRs each column separately.
_MIN_GUTTER_WIDTH_RATIO = 0.03        # gutter must be >= 3% of image width to count
_GUTTER_SEARCH_RANGE = (0.25, 0.75)   # only look for a gutter in the middle half of
                                       # the page, to avoid false positives from
                                       # ordinary left/right page margins


class ImageParseError(Exception):
    """Raised when an image cannot be parsed or OCR'd."""
    pass


def _fix_ocr_digit_errors(text: str) -> str:
    """Corrects common Tesseract digit misreads (0->@/o/e, 9->g) inside
    tokens that are mostly numeric (IDs, row numbers, amounts)."""
    _SUBS = {'@': '0', 'e': '0', 'g': '9'}

    def fix_token(m):
        tok = m.group(0)
        digit_ratio = sum(c.isdigit() for c in tok) / len(tok)
        if digit_ratio < 0.4:
            return tok
        return "".join(_SUBS.get(c, c) for c in tok)

    return re.sub(r"\S+", fix_token, text)


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    General-purpose preprocessing to improve OCR accuracy on any image
    type (prose, tables, forms) — not tailored to any specific document.
    """
    img = ImageOps.grayscale(img)

    if img.width < _MIN_OCR_WIDTH:
        scale = _MIN_OCR_WIDTH / img.width
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def _otsu_threshold(histogram: list[int]) -> int:
    """
    Compute the Otsu threshold from a 256-bin grayscale histogram.
    Pure-Python implementation — no numpy/opencv dependency.
    """
    total = sum(histogram)
    sum_total = sum(i * histogram[i] for i in range(256))

    sum_bg, weight_bg, max_variance, threshold = 0.0, 0, 0.0, 0

    for i in range(256):
        weight_bg += histogram[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += i * histogram[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = i

    return threshold


def _binarize_otsu(img: Image.Image) -> Image.Image:
    """
    Threshold a grayscale image to pure black/white using an
    automatically-computed (Otsu) threshold. Removes midtone noise
    that otherwise confuses character-edge recognition — one of the
    highest-impact preprocessing steps for screenshots/rendered text.
    """
    histogram = img.histogram()
    threshold = _otsu_threshold(histogram)
    return img.point(lambda p: 255 if p > threshold else 0)


def _mean_confidence(img: Image.Image, config: str) -> tuple[str, float]:
    """
    Run OCR with the given config and return (text, mean_word_confidence).
    Confidence values of -1 (no text detected for that box) are excluded.
    """
    data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    confidences = [int(c) for c in data["conf"] if c not in ("-1", -1)]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    text = pytesseract.image_to_string(img, config=config).strip()
    return text, mean_conf


def _detect_column_gutter(img: Image.Image, config: str) -> int | None:
    """
    Looks for a wide vertical gap with no word content in the middle
    portion of the page — a strong signal of a two-column layout (e.g.
    Seller/Client fields side by side on an invoice). Returns the
    x-coordinate to split the image on, or None if no clear gutter is
    found (single-column / prose documents won't have one — that's the
    expected common case, and this function should stay a no-op for them).
    """
    data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)

    word_boxes = [
        (data["left"][i], data["left"][i] + data["width"][i])
        for i in range(len(data["text"]))
        if data["text"][i].strip()
    ]
    if not word_boxes:
        return None

    width = img.width
    search_start = int(width * _GUTTER_SEARCH_RANGE[0])
    search_end = int(width * _GUTTER_SEARCH_RANGE[1])
    min_gutter_width = int(width * _MIN_GUTTER_WIDTH_RATIO)

    if search_end <= search_start:
        return None

    # Build a simple "coverage" array across the search zone: mark every
    # x-pixel that falls inside a word's bounding box as occupied.
    occupied = [False] * (search_end - search_start)
    for left, right in word_boxes:
        lo = max(left, search_start)
        hi = min(right, search_end)
        for x in range(lo, hi):
            occupied[x - search_start] = True

    # Find the widest contiguous run of unoccupied pixels in the search zone.
    best_run_start, best_run_len = None, 0
    run_start, run_len = None, 0
    for i, is_occupied in enumerate(occupied):
        if not is_occupied:
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            if run_len > best_run_len:
                best_run_len, best_run_start = run_len, run_start
            run_start, run_len = None, 0
    if run_len > best_run_len:
        best_run_len, best_run_start = run_len, run_start

    if best_run_start is None or best_run_len < min_gutter_width:
        return None

    gutter_center = search_start + best_run_start + best_run_len // 2
    logger.info(f"Detected column gutter at x={gutter_center} (width={best_run_len}px)")
    return gutter_center


def _ocr_two_column(img: Image.Image, gutter_x: int, config: str) -> str:
    """OCR each column independently and concatenate — preserves the
    correct reading order within each column instead of interleaving
    both columns line by line."""
    left_col = img.crop((0, 0, gutter_x, img.height))
    right_col = img.crop((gutter_x, 0, img.width, img.height))

    left_text = pytesseract.image_to_string(left_col, config=config).strip()
    right_text = pytesseract.image_to_string(right_col, config=config).strip()

    return f"{left_text}\n\n{right_text}"


def parse_image(file_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.

    Args:
        file_path: Path to the image file.

    Returns:
        Extracted text.

    Raises:
        ImageParseError: If the file doesn't exist, isn't a supported
                          image format, or no text could be extracted.
    """
    path = Path(file_path)

    if not path.exists():
        raise ImageParseError(f"File not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageParseError(
            f"Unsupported image format: {path.suffix} "
            f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )

    try:
        img = Image.open(file_path)
        img.load()  # force-read the file now, catches truncated/corrupt images early
    except UnidentifiedImageError:
        raise ImageParseError(f"File is not a valid image (corrupted or wrong format): {file_path}")
    except Exception as e:
        raise ImageParseError(f"Could not open image: {e}")

    img = _preprocess_for_ocr(img)

    try:
        text, confidence = _mean_confidence(img, _OCR_CONFIG_PRIMARY)

        if confidence < _CONFIDENCE_RETRY_THRESHOLD:
            fallback_text, fallback_confidence = _mean_confidence(img, _OCR_CONFIG_FALLBACK)
            logger.info(
                f"{path.name}: primary OCR confidence {confidence:.1f} below threshold, "
                f"fallback scored {fallback_confidence:.1f}"
            )
            if fallback_confidence > confidence:
                text = fallback_text

        # Two-column detection: run regardless of confidence, since a
        # merged two-column layout can still score "confident" per-word
        # even though the resulting reading order is wrong (each word is
        # read correctly, they're just interleaved from two columns).
        try:
            gutter_x = _detect_column_gutter(img, _OCR_CONFIG_PRIMARY)
        except Exception as e:
            logger.warning(f"{path.name}: column gutter detection failed, skipping ({e})")
            gutter_x = None

        if gutter_x is not None:
            try:
                column_text = _ocr_two_column(img, gutter_x, _OCR_CONFIG_PRIMARY)
                if len(column_text.strip()) >= MIN_TEXT_LENGTH:
                    logger.info(f"{path.name}: two-column layout detected, using column-split OCR")
                    text = column_text
            except Exception as e:
                logger.warning(f"{path.name}: column-split OCR failed, keeping single-pass result ({e})")

    except pytesseract.TesseractNotFoundError:
        raise ImageParseError(
            "Tesseract OCR engine is not installed or not found in PATH. "
            "Install it separately from the pytesseract Python package "
            "(see: https://github.com/UB-Mannheim/tesseract/wiki for Windows)."
        )
    except Exception as e:
        raise ImageParseError(f"OCR failed: {e}")

    text = _fix_ocr_digit_errors(text)

    if len(text) < MIN_TEXT_LENGTH:
        raise ImageParseError(
            f"No readable text found in {file_path}. "
            f"This may be a non-text image (photo, complex diagram, or blank)."
        )
    OCR_DEBUG_LOG.append({
        "filename": path.name,
        "char_count": len(text),
        "pipe_count": text.count("|"),
        "line_count": text.count("\n") + 1,
        "raw_text": text,
    })

    logger.info(f"Extracted {len(text)} characters from {path.name} via OCR")
    return text