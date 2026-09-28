"""Render font/point-size calibration pages from synthetic or user-provided text."""
import argparse
import os
import json
import random
import re
import sys
from pathlib import Path

from pdf2image import convert_from_bytes
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate
from xml.sax.saxutils import escape
import io

from acuity_lib import FONTS, load_fonts

PAGE_SIZE = (595.0, 842.0)
MARGIN = 10.0
DPI = 72

_env_bin = Path(sys.executable).parent
POPPLER = str(_env_bin) if (_env_bin / "pdftoppm").exists() else None
_WS = re.compile(r"\s+")


def build_pdf(text, font_path, size):
    # 名字必须按字体唯一：reportlab 的字体表按名字缓存，重复注册同名会被忽略，
    # 于是一个进程里所有字体都会用第一个注册的那套字形。
    name = "probe_" + re.sub(r"\W+", "_", font_path)
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(name, font_path))
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=PAGE_SIZE, leftMargin=MARGIN,
                            rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)
    style = ParagraphStyle(name=name, parent=getSampleStyleSheet()["Normal"],
                           fontName=name, fontSize=size, leading=size + 1.0,
                           textColor=colors.black, alignment=TA_LEFT)

    def bg(canvas, _doc):
        canvas.saveState()
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, *PAGE_SIZE, stroke=0, fill=1)
        canvas.restoreState()

    doc.build([Paragraph(escape(text), style)], onFirstPage=bg, onLaterPages=bg)
    out = buf.getvalue()
    buf.close()
    return out


def natural_passages(n, words, source):
    """Read a user-provided JSONL corpus with one `context` field per row."""
    rows = []
    with Path(source).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line)["context"])
            if len(rows) >= 400:
                break
    rng = random.Random(11)
    rng.shuffle(rows)
    out, seen = [], set()
    for ctx in rows:
        ctx = ctx.replace("NEWLINE_CHAR", " ")
        ctx = "".join(c for c in ctx if c.isascii() and (c.isprintable() or c == " "))
        toks = _WS.sub(" ", ctx).strip().split(" ")
        if len(toks) < words * 3:
            continue
        start = rng.randrange(words, len(toks) - words)
        p = " ".join(toks[start:start + words])
        if p[:40] in seen:
            continue
        seen.add(p[:40])
        out.append(p)
        if len(out) == n:
            break
    if len(out) < n:
        raise ValueError(f"Found only {len(out)} suitable passages; requested {n}")
    return out


def random_passages(n, words):
    """随机字母数字串，掐掉语言模型先验。词长和大小写/数字比例照英文语料。"""
    rng = random.Random(23)
    low = "abcdefghijklmnopqrstuvwxyz"
    out = []
    for _ in range(n):
        ws = []
        for _ in range(words):
            L = rng.choice([3, 4, 4, 5, 5, 6, 6, 7, 8])
            w = "".join(rng.choice(low) for _ in range(L))
            r = rng.random()
            if r < 0.06:
                w = "".join(rng.choice("0123456789") for _ in range(L))
            elif r < 0.12:
                w = w.capitalize()
            ws.append(w)
        out.append(" ".join(ws))
    return out


def confusable_passages(n, words):
    """Sample confusable character groups at elevated frequency."""
    units = ["cl", "d", "rn", "m", "vv", "w", "il", "u", "ye", "4", "9", "0", "o",
             "c", "l", "i", "j", "q", "g", "a", "t", "f", "s", "x", "k", "1", "7"]
    rng = random.Random(31)
    out = []
    for _ in range(n):
        ws = []
        for _ in range(words):
            w = ""
            while len(w) < rng.choice([4, 5, 5, 6, 6, 7]):
                w += rng.choice(units)
            ws.append(w)
        out.append(" ".join(ws))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("FOCUSVTC_ACUITY_ROOT", "outputs/font_acuity"))
    ap.add_argument("--sizes", type=float, nargs="+", default=[4, 5, 6, 7, 8, 9, 11, 13])
    ap.add_argument("--fonts", nargs="+", default=["dejavu"])
    ap.add_argument("--conds", nargs="+", choices=["natural", "random", "confusable"], default=["random", "confusable"])
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--words-natural", type=int, default=90)
    ap.add_argument("--words-random", type=int, default=60)
    ap.add_argument("--font-config", help="JSON mapping of font labels to .ttf paths")
    ap.add_argument("--natural-source", help="JSONL corpus with a context field per row")
    a = ap.parse_args()
    if a.n < 1 or a.words_natural < 1 or a.words_random < 1 or any(s <= 0 for s in a.sizes):
        ap.error("Counts and point sizes must be positive")
    if "natural" in a.conds and not a.natural_source:
        ap.error("--natural-source is required for the natural condition")
    load_fonts(a.font_config)
    for font in a.fonts:
        if font not in FONTS or not Path(FONTS[font]).is_file():
            ap.error(f"Font {font!r} is missing; supply --font-config or FOCUSVTC_FONT_PATH")

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    builders = {"natural": lambda: natural_passages(a.n, a.words_natural, a.natural_source),
                "random": lambda: random_passages(a.n, a.words_random),
                "confusable": lambda: confusable_passages(a.n, a.words_random)}
    corpora = {k: builders[k]() for k in a.conds}
    # 合并而不是覆盖：分多次按 cond 渲染时，直接写会把上一次的语料抹掉
    pj = out / "passages.json"
    merged = json.loads(pj.read_text()) if pj.exists() else {}
    merged.update(corpora)
    pj.write_text(json.dumps(merged, ensure_ascii=False, indent=1))

    manifest = []
    for cond, texts in corpora.items():
        for font in a.fonts:
            for size in a.sizes:
                d = out / f"{cond}__{font}__pt{size:g}"
                d.mkdir(exist_ok=True)
                for i, text in enumerate(texts):
                    png = d / f"{i:03d}.png"
                    pages = convert_from_bytes(build_pdf(text, FONTS[font], size),
                                               dpi=DPI, poppler_path=POPPLER)
                    if len(pages) != 1:
                        raise ValueError("Calibration passage spans pages; reduce words per passage")
                    pages[0].save(png, "PNG")
                    manifest.append({"cond": cond, "font": font, "pt": size, "idx": i,
                                     "image": str(png), "gold": text})
                print(f"  {cond:8s} {font:8s} {size:>4g}pt  {len(texts)} 页", file=sys.stderr)

    (out / a.manifest).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                          for r in manifest))
    print(f"\n共 {len(manifest)} 张 -> {out}/{a.manifest}")


if __name__ == "__main__":
    main()
