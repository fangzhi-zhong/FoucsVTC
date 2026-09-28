"""Render REL-CoT text and move existing evidence through its character stream."""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
import re
import tempfile
from typing import NamedTuple
import unicodedata
from xml.sax.saxutils import escape


PAGE_SIZE = (595.0, 842.0)
MARGIN_PT = 10.0
FONT_SIZE_PT = 9.0
LINE_HEIGHT_PT = 10.0
PAGE_CHUNK_SIZE = 8
# Error is normalized by ink, so a mostly blank page cannot hide a wrong font.
# Minor grayscale rasterization differences are allowed; shifted text is not.
MAX_RELATIVE_INK_ERROR = 0.15
EVIDENCE_PADDING_PT = 2.0


class _Glyph(NamedTuple):
    page: int
    left: float
    top: float
    right: float
    bottom: float


class _TextLayout(NamedTuple):
    text: str
    glyphs: list[_Glyph]
    page_ranges: dict[int, range]


def _register_output_font(font_path: Path) -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font = TTFont(str(font_path), str(font_path))
    family = font.face.familyName
    if isinstance(family, bytes):
        family = family.decode("utf-8", errors="replace")
    if " ".join(str(family).split()).casefold() != "dejavu sans":
        raise ValueError(f"Output font must be DejaVu Sans; {font_path} has family {family!r}")
    pdfmetrics.registerFont(font)


def _build_pdf(text: str, font_path: Path) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    font_name = str(font_path)
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN_PT,
        rightMargin=MARGIN_PT,
        topMargin=MARGIN_PT,
        bottomMargin=MARGIN_PT,
    )
    options = {
        "name": "RELCoT",
        "parent": getSampleStyleSheet()["Normal"],
        "fontName": font_name,
        "fontSize": FONT_SIZE_PT,
        "leading": LINE_HEIGHT_PT,
        "textColor": colors.black,
        "backColor": colors.white,
        "alignment": TA_LEFT,
    }
    if re.search(r"[\u4e00-\u9fff]", text):
        options["wordWrap"] = "CJK"
    style = ParagraphStyle(**options)
    cleaned = text.replace("\xad", "").replace("\u200b", "")
    processed = re.sub(r" {2,}", lambda match: "&nbsp;" * len(match.group()), escape(cleaned))
    processed = processed.replace("\n", "<br/>").replace("\t", "&nbsp;" * 4)

    def draw_background(canvas, _document):
        canvas.saveState()
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, PAGE_SIZE[0], PAGE_SIZE[1], stroke=0, fill=1)
        canvas.restoreState()

    document.build(
        [Paragraph(processed, style)],
        onFirstPage=draw_background,
        onLaterPages=draw_background,
    )
    return buffer.getvalue()


def _page_chunks(pdf: bytes, dpi: int, count: int, poppler_path: str | None):
    from pdf2image import convert_from_bytes

    for first in range(1, count + 1, PAGE_CHUNK_SIZE):
        last = min(count, first + PAGE_CHUNK_SIZE - 1)
        pages = convert_from_bytes(
            pdf,
            dpi=dpi,
            first_page=first,
            last_page=last,
            poppler_path=poppler_path,
            thread_count=1,
        )
        try:
            if len(pages) != last - first + 1:
                raise ValueError(f"Poppler returned an incomplete page range {first}-{last}")
            yield first - 1, pages
        finally:
            for page in pages:
                page.close()


def _reference_crop(page, reference_size: tuple[int, int], dpi: int):
    width, height = (max(1, round(size * dpi / 72.0)) for size in reference_size)
    if width > page.width or height > page.height:
        raise ValueError(
            f"72-DPI reference geometry {reference_size} exceeds the PDF page "
            f"at {dpi} DPI ({page.width}x{page.height})"
        )
    return page.crop((0, 0, width, height))


def _check_layout(pdf: bytes, references: list[Path], sizes, poppler_path: str | None) -> None:
    import numpy as np
    from PIL import Image

    # Explicitly close the chunk iterator when a font fails partway through.
    chunks = _page_chunks(pdf, 72, len(references), poppler_path)
    try:
        for start, pages in chunks:
            for offset, page in enumerate(pages):
                index = start + offset
                with _reference_crop(page, sizes[index], 72) as cropped:
                    candidate = np.asarray(cropped.convert("L"), dtype=np.float32)
                with Image.open(references[index]) as reference:
                    original = np.asarray(reference.convert("L"), dtype=np.float32)
                ink_mass = float((255.0 - original).sum())
                error = float(np.abs(candidate - original).sum()) / max(ink_mass, 1.0)
                if error > MAX_RELATIVE_INK_ERROR:
                    raise ValueError(
                        f"page {index + 1}: 72-DPI text does not match the reference "
                        f"(relative ink error {error:.3f})"
                    )
    finally:
        chunks.close()


def _save_png(image, path: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            if isinstance(image, bytes):
                stream.write(image)
            else:
                image.save(stream, format="PNG")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _text_layout(
    pdf: bytes,
    *,
    source_text: str | None = None,
    font_path: Path | None = None,
) -> _TextLayout:
    import pdfplumber

    letters, glyphs, page_ranges = [], [], {}
    with pdfplumber.open(io.BytesIO(pdf)) as document:
        for number, page in enumerate(document.pages, 1):
            start = len(glyphs)
            try:
                for char in page.chars:
                    glyph = _Glyph(
                        number, float(char["x0"]), float(char["top"]),
                        float(char["x1"]), float(char["bottom"]),
                    )
                    for letter in char["text"]:
                        if not letter.isspace():
                            letters.append(letter)
                            glyphs.append(glyph)
                page_ranges[number] = range(start, len(glyphs))
            finally:
                page.close()

    if source_text is not None:
        from reportlab.pdfbase import pdfmetrics

        if font_path is None:
            raise ValueError("The PDF font is required to align its source text")
        cleaned = source_text.replace("\xad", "").replace("\u200b", "")
        expected = [letter for letter in cleaned if not letter.isspace()]
        if len(letters) != len(expected):
            raise ValueError(
                "The PDF and source text have different non-whitespace character "
                f"counts ({len(letters)} and {len(expected)}); evidence cannot be aligned"
            )
        char_to_glyph = pdfmetrics.getFont(str(font_path)).face.charToGlyph
        for index, (actual, original) in enumerate(zip(letters, expected, strict=True)):
            if actual == original:
                continue
            # ReportLab encodes characters absent from a font as glyph zero;
            # pdfplumber consequently extracts NUL. Recover only a source
            # character that this exact font's cmap proves is missing. The
            # character count and every other position still have to match.
            missing_glyph = actual == "\x00" and not char_to_glyph.get(ord(original), 0)
            equivalent = unicodedata.normalize("NFKC", actual) == unicodedata.normalize("NFKC", original)
            if not missing_glyph and not equivalent:
                raise ValueError(
                    f"PDF/source character mismatch at {index}: {actual!r} versus "
                    f"{original!r}; evidence cannot be aligned"
                )
            letters[index] = original

    normalized_letters, normalized_glyphs, normalized_ranges = [], [], {}
    for page, indices in page_ranges.items():
        start = len(normalized_glyphs)
        for index in indices:
            for letter in unicodedata.normalize("NFKC", letters[index]):
                if not letter.isspace():
                    normalized_letters.append(letter)
                    normalized_glyphs.append(glyphs[index])
        normalized_ranges[page] = range(start, len(normalized_glyphs))
    return _TextLayout("".join(normalized_letters), normalized_glyphs, normalized_ranges)


def _target_geometry(pdf: bytes, count: int, poppler_path: str | None, cache_72: bool):
    """Use the published right/last-bottom crop rule once, at the new 72 DPI."""
    import numpy as np

    sizes, encoded_pages = [], []
    chunks = _page_chunks(pdf, 72, count, poppler_path)
    try:
        for start, pages in chunks:
            for offset, page in enumerate(pages):
                gray = np.asarray(page.convert("L"), dtype=np.float32)
                background = float(np.median(gray[:2, :2]))
                ink = np.abs(gray - background) > 5
                width, height = page.size
                columns = np.where(ink.any(axis=0))[0]
                if columns.size:
                    width = min(width, int(columns[-1] + 1 + MARGIN_PT))
                if start + offset == count - 1:
                    rows = np.where(ink.any(axis=1))[0]
                    if rows.size:
                        height = min(height, int(rows[-1] + MARGIN_PT))
                sizes.append((width, height))
                if cache_72:
                    with page.crop((0, 0, width, height)) as cropped:
                        buffer = io.BytesIO()
                        cropped.save(buffer, format="PNG")
                        encoded_pages.append(buffer.getvalue())
    finally:
        chunks.close()
    return sizes, encoded_pages


def _location_values(location, page_count):
    page, bbox = location.get("page"), location.get("bbox")
    if isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= page_count:
        raise ValueError(f"Evidence page is outside the source document: {page!r}")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Page {page}: evidence bbox must have four coordinates")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in bbox):
        raise ValueError(f"Page {page}: evidence coordinates must be finite numbers")
    left, top, right, bottom = bbox
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        raise ValueError(f"Page {page}: invalid normalized evidence box {bbox}")
    return page, bbox


def _move_locations(locations, source_layout, target_layout, source_sizes, target_sizes):
    if source_layout.text != target_layout.text:
        common = min(len(source_layout.text), len(target_layout.text))
        mismatch = next(
            (index for index in range(common)
             if source_layout.text[index] != target_layout.text[index]),
            common,
        )
        raise ValueError(
            "The source and output PDFs have different non-whitespace NFKC character "
            f"streams (first mismatch at {mismatch}; lengths "
            f"{len(source_layout.text)} and {len(target_layout.text)}). "
            "Evidence cannot be transferred safely; check the source text and fonts."
        )
    if not source_layout.text:
        raise ValueError("The source PDF has no readable characters to align")

    page_sets = {page: set() for page in source_layout.page_ranges}
    for source, target in zip(source_layout.glyphs, target_layout.glyphs, strict=True):
        page_sets[source.page].add(target.page)
    page_map = {page: sorted(targets) for page, targets in page_sets.items()}

    location_map = []
    for location in locations:
        page, bbox = _location_values(location, len(source_layout.page_ranges))
        left, top, right, bottom = bbox
        width, height = source_sizes[page - 1]
        x0, y0 = left * width / 1000, top * height / 1000
        x1, y1 = right * width / 1000, bottom * height / 1000
        targets = {}
        for index in source_layout.page_ranges[page]:
            source = source_layout.glyphs[index]
            if (x0 <= (source.left + source.right) / 2 <= x1
                    and y0 <= (source.top + source.bottom) / 2 <= y1):
                target = target_layout.glyphs[index]
                targets.setdefault(target.page, set()).add(target)
        if not targets:
            raise ValueError(f"Page {page}: evidence box {bbox} contains no source character centers")
        mapped = []
        for target_page, glyphs in sorted(targets.items()):
            width, height = target_sizes[target_page - 1]
            left = max(0.0, min(glyph.left for glyph in glyphs) - EVIDENCE_PADDING_PT)
            top = max(0.0, min(glyph.top for glyph in glyphs) - EVIDENCE_PADDING_PT)
            right = min(float(width), max(glyph.right for glyph in glyphs) + EVIDENCE_PADDING_PT)
            bottom = min(float(height), max(glyph.bottom for glyph in glyphs) + EVIDENCE_PADDING_PT)
            if left >= right or top >= bottom:
                raise ValueError(f"Page {target_page}: mapped evidence lies outside the output crop")
            mapped.append({
                "page": target_page,
                "bbox": [
                    math.floor(left / width * 1000), math.floor(top / height * 1000),
                    math.ceil(right / width * 1000), math.ceil(bottom / height * 1000),
                ],
            })
        location_map.append({"source": dict(location), "targets": mapped})
    return location_map, page_map


def render_sample(
    text: str,
    reference_images: list[str],
    output_dirs: dict[int, str],
    font_path: str,
    *,
    source_font_paths: list[str] | None = None,
    poppler_path: str | None = None,
    locations: list[dict] | None = None,
) -> dict:
    """Render all requested DPIs in one output font and transfer source evidence.

    ``reference_images`` must be the ordered, original 72-DPI pages. Candidate
    source fonts begin with ``font_path`` and continue with ``source_font_paths``;
    they only reconstruct the old layout. All outputs, including 72 DPI, use
    ``font_path``. Character alignment maps evidence and page references across
    changed line wrapping and pagination. The new PDF's 72-DPI crop determines
    the crop geometry at every DPI. ``views`` uses integer DPI keys.
    """
    from pdf2image import pdfinfo_from_bytes
    from PIL import Image
    from reportlab.pdfbase.ttfonts import TTFError
    from reportlab.platypus.doctemplate import LayoutError

    if not isinstance(text, str) or not text.strip():
        raise ValueError("REL-CoT source text is empty")
    if not reference_images:
        raise ValueError("REL-CoT requires the original 72-DPI reference images")
    output_font = Path(font_path).expanduser().resolve()
    if not output_font.is_file():
        raise FileNotFoundError(f"Output font file is missing: {output_font}")
    _register_output_font(output_font)
    if any(isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0 for dpi in output_dirs):
        raise ValueError("Output DPIs must be positive integers")

    references = [Path(path).expanduser().resolve() for path in reference_images]
    sizes = []
    for path in references:
        with Image.open(path) as image:
            sizes.append(image.size)
    directories = {dpi: Path(path).expanduser().resolve() for dpi, path in output_dirs.items()}
    if len(set(directories.values())) != len(directories):
        raise ValueError("Each output DPI needs a separate directory")

    failures = []
    selected_font = None
    selected_pdf = None
    seen_fonts = set()
    for value in [str(output_font), *(source_font_paths or [])]:
        source_font = Path(value).expanduser().resolve()
        if source_font in seen_fonts:
            continue
        seen_fonts.add(source_font)
        if not source_font.is_file():
            failures.append(f"{source_font}: font file is missing")
            continue
        try:
            pdf = _build_pdf(text, source_font)
            count = int(pdfinfo_from_bytes(pdf, poppler_path=poppler_path)["Pages"])
            if count != len(references):
                raise ValueError(f"PDF has {count} pages; the 72-DPI reference has {len(references)}")
            _check_layout(pdf, references, sizes, poppler_path)
        except (ValueError, OSError, TTFError, LayoutError) as error:
            failures.append(f"{source_font}: {error}")
            continue
        selected_font, selected_pdf = source_font, pdf
        break
    if selected_font is None:
        raise ValueError(
            "No candidate font reproduces the original 72-DPI REL-CoT layout. "
            "Provide the font used by the reference pages in source_font_paths so their "
            "evidence can be mapped to the output font. " + "; ".join(failures)
        )

    if selected_font == output_font:
        # A matching DejaVu reference already defines the correct annotation
        # geometry. Retain its boxes and crop rather than tightening them.
        target_pdf, target_sizes, target_count = selected_pdf, sizes, len(references)
        cached_72 = None
        page_map = {page: [page] for page in range(1, target_count + 1)}
        location_map = []
        for location in locations or []:
            page, bbox = _location_values(location, target_count)
            location_map.append({
                "source": dict(location),
                "targets": [{"page": page, "bbox": list(bbox)}],
            })
    else:
        target_pdf = _build_pdf(text, output_font)
        source_layout = _text_layout(selected_pdf, source_text=text, font_path=selected_font)
        target_layout = _text_layout(target_pdf, source_text=text, font_path=output_font)
        target_count = len(target_layout.page_ranges)
        target_sizes, cached_72 = _target_geometry(target_pdf, target_count, poppler_path, 72 in directories)
        location_map, page_map = _move_locations(
            locations or [], source_layout, target_layout, sizes, target_sizes,
        )

    views = {}
    for dpi, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        paths, image_sizes = [], []
        if dpi == 72 and cached_72 is not None:
            for index, encoded in enumerate(cached_72):
                path = directory / f"page_{index + 1:03d}.png"
                _save_png(encoded, path)
                paths.append(str(path))
            views[dpi] = {"image": paths, "image_sizes": [list(size) for size in target_sizes], "num_pages": len(paths)}
            continue
        chunks = _page_chunks(target_pdf, dpi, target_count, poppler_path)
        try:
            for start, pages in chunks:
                for offset, page in enumerate(pages):
                    index = start + offset
                    path = directory / f"page_{index + 1:03d}.png"
                    with _reference_crop(page, target_sizes[index], dpi) as cropped:
                        _save_png(cropped, path)
                        image_sizes.append(list(cropped.size))
                    paths.append(str(path))
        finally:
            chunks.close()
        views[dpi] = {"image": paths, "image_sizes": image_sizes, "num_pages": len(paths)}
    return {
        "views": views,
        "font_path": str(output_font),
        "source_font_path": str(selected_font),
        "location_map": location_map,
        "page_map": page_map,
    }
