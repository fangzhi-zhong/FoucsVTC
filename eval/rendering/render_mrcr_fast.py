#!/usr/bin/env python3
"""Fast entry point for the MRCR VTC renderer.

It reuses ``render_mrcr.py`` but lays out bounded groups of text lines instead
of one multi-million-character ReportLab Paragraph.
"""

from pathlib import Path
from xml.sax.saxutils import escape

if __package__:
    from . import render_mrcr as base
else:
    import render_mrcr as base


def build_pdf_fast(text: str, font_key: str, path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    font_name = base.register_font(font_key)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=base.PAGE_SIZE,
        leftMargin=base.MARGIN_X,
        rightMargin=base.MARGIN_X,
        topMargin=base.MARGIN_Y,
        bottomMargin=base.MARGIN_Y,
    )
    style = ParagraphStyle(
        name="MRCRVTCFast",
        parent=getSampleStyleSheet()["Normal"],
        fontName=font_name,
        fontSize=base.FONT_SIZE,
        leading=base.LINE_HEIGHT,
        textColor=colors.HexColor(base.FONT_COLOR),
        backColor=colors.HexColor(base.PAGE_BG),
        alignment=TA_LEFT,
        wordWrap="CJK" if font_key == "cjk" else None,
        spaceBefore=0,
        spaceAfter=0,
    )

    def markup(block: str) -> str:
        block = block.replace("\xad", "").replace("\u200b", "")
        body = base._RE_MULTISPACE.sub(
            lambda match: "&nbsp;" * len(match.group()), escape(block)
        )
        return body.replace("\n", "<br/>").replace("\t", "&nbsp;" * 4)

    def draw_bg(canvas, _doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(base.PAGE_BG))
        canvas.rect(0, 0, base.PAGE_SIZE[0], base.PAGE_SIZE[1], stroke=0, fill=1)
        canvas.restoreState()

    # Each block is small enough for near-linear layout. A new zero-spacing
    # Paragraph supplies the line break removed at each block boundary.
    lines = text.split("\n")
    blocks = ["\n".join(lines[start : start + 32]) for start in range(0, len(lines), 32)]
    story = [Paragraph(markup(block) or "&nbsp;", style) for block in blocks]
    doc.build(story, onFirstPage=draw_bg, onLaterPages=draw_bg)


if __name__ == "__main__":
    base.RENDER_VERSION = 2
    base.build_pdf = build_pdf_fast
    base.main()
