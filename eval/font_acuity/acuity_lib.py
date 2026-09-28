"""Font calibration metrics, threshold interpolation, and bootstrap intervals."""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import ImageFont

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FOCUSVTC_ACUITY_ROOT", "outputs/font_acuity"))
# Character error rate threshold and font measurement resolution.
CRIT = 0.05
HI = 2048
FONTS = {"dejavu": os.environ.get(
    "FOCUSVTC_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")}


def load_fonts(path=None):
    """Load a user-owned JSON mapping of font labels to TrueType font paths."""
    if path:
        config = Path(path).resolve()
        mapping = json.loads(config.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Font configuration must be a nonempty JSON object")
        for label, value in mapping.items():
            font_path = Path(os.path.expandvars(value)).expanduser()
            if not font_path.is_absolute():
                font_path = config.parent / font_path
            FONTS[label] = str(font_path)
    return FONTS


def load(files=None):
    """-> {(cond, font, pt): {idx: cer}}，CER 截到 1.0（插入过多时会超过 1）。

    同一组合出现多行时取平均：vLLM 即使 temperature=0 也会因连续批处理的批次构成不同
    产生小幅数值抖动，重复测量是独立样本，平均比丢弃更好。
    """
    if files is None:
        files = tuple(x for x in os.environ.get(
            "FONT_ACUITY_FILES", "outputs/font_acuity/transcribe.jsonl").split(",") if x)
    acc = defaultdict(lambda: defaultdict(list))
    for name in files:
        p = Path(name)
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l)
            # Transport/API failures are not visual errors.  New result files mark
            # them with ``ok=false`` and ``cer=null``; keep legacy files compatible.
            if r.get("ok", True) is False or r.get("cer") is None:
                continue
            acc[(r["cond"], r["font"], r["pt"])][r["idx"]].append(min(r["cer"], 1.0))
    return {k: {i: float(np.mean(v)) for i, v in d.items()} for k, d in acc.items()}


def curve(per, cond, font, pts, idxs=None):
    if idxs is None:
        idxs = sorted(set.intersection(*(set(per.get((cond, font, p), {})) for p in pts))) if pts else []
    if len(idxs) == 0:
        return [np.nan for _ in pts]
    return [float(np.mean([per[(cond, font, p)][i] for i in idxs])) for p in pts]


def interp(pts, mean, crit=CRIT):
    a = np.asarray(mean)
    x = np.log(np.asarray(pts, dtype=float))
    for i in range(len(a) - 1):
        if a[i] == crit:
            return float(pts[i])
        if a[i] >= crit >= a[i + 1]:
            t = (a[i] - crit) / (a[i] - a[i + 1])
            return float(np.exp(x[i] + t * (x[i + 1] - x[i])))
    return float(pts[-1]) if len(a) and a[-1] == crit else np.nan


def bootstrap(per, cond, font, pts, crit=CRIT, n=3000, seed=1):
    """重采样段落，给阈值的 95% CI。段落难度是主要的方差来源。"""
    idxs = sorted(set.intersection(*(set(per.get((cond, font, p), {})) for p in pts))) if pts else []
    if not idxs:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = interp(pts, curve(per, cond, font, pts, rng.choice(idxs, len(idxs))), crit)
        if np.isfinite(v):
            out.append(v)
    return tuple(np.percentile(out, [2.5, 97.5])) if len(out) > 50 else (np.nan, np.nan)


def constants(font_key, corpus):
    """在实测语料上量平均字宽 a（/em）和 x 高比 r（/em）。"""
    f = ImageFont.truetype(FONTS[font_key], HI)
    text = " ".join(corpus)
    box = f.getbbox("x")
    return f.getlength(text) / len(text) / HI, (box[3] - box[1]) / HI


def cost(a, E):
    """阈值处每字符吃掉的页面像素：字宽 x 行高（渲染时 leading = 字号 + 1）。"""
    return a * E * (E + 1.0)


def passages(cond="random"):
    return json.load(open(DATA / "passages.json"))[cond]
