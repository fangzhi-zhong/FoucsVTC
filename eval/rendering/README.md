# Evaluation-data rendering

These scripts render locally supplied benchmark text into page images and
preserve the questions and reference answers needed by evaluation. Source
datasets, images, and font binaries are not included. REL-CoT training-data
rendering and SFT format conversion are documented separately in
[`data/`](../../data/README.md). The tools reuse existing supervision;
automatic REL annotation and synthetic training-task generation are not
included.

| Input | Entry point | Output |
| --- | --- | --- |
| A UTF-8 document | `render_document.py` | Pages and `manifest.json` |
| RULER v1 | `render_ruler_v1.py` | `RULER_v1_VTC/dpi_<dpi>/index.jsonl` |
| RULER v2 | `render_ruler_v2.py` | `RULER_v2_VTC/dpi_<dpi>/<task>/test.jsonl` |
| LongBench | `render_longbench.py` | `LongBench_VTC/dpi_<dpi>/<dataset>/test.jsonl` |
| MRCR | `render_mrcr_fast.py` / `render_mrcr.py` | `MRCR_VTC/dpi_<dpi>/<subset>/test.jsonl` |
| VTCBench-Wild | `render_vtcbench.py` | `VTCBench-Wild_VTC/dpi_<dpi>/<tier>/<split>-00000-of-00001.json` |

MMMU and OCRBench use dataset-provided images, so they have no text renderer.
Font and point-size calibration stays under
[`eval/font_acuity/`](../font_acuity/README.md).

## Setup

Run the commands below from the FocusVTC repository root with Python 3.12:

```bash
python -m pip install -r eval/rendering/requirements.txt
export FOCUSVTC_DATA_ROOT="$PWD/datasets"
export FOCUSVTC_FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
```

The benchmark renderers use `pdf2image` and require the system Poppler tools
(`pdftoppm` and `pdfinfo`). The standalone document renderer uses PDFium and
needs no Poppler installation.

The main font defaults to DejaVu Sans at the system path above, overridden by
`FOCUSVTC_FONT_PATH` or an explicit `--font-path`. Install the font locally;
no font binaries are bundled. The page layout uses a 9 pt font, 10 pt line
height, 595 × 842 pt pages, and 10 pt margins. Changing fonts changes
pagination and visual difficulty.

DejaVu Sans does not cover all CJK characters. For text with unsupported
characters, supply a suitable font using `--cjk-font-path` or
`FOCUSVTC_CJK_FONT_PATH` in LongBench, MRCR, or VTCBench. LongBench also accepts
`--mono-font-path` or `FOCUSVTC_MONO_FONT_PATH` for explicit monospace code
rendering. If these overrides are omitted, all font roles use the main font.

RULER v2, LongBench, MRCR, and VTCBench default to sources under
`$FOCUSVTC_DATA_ROOT/sources/{ruler_v2,longbench,mrcr,vtcbench}` and the output
roots listed above. If unset, the data root is the repository's `datasets/`
directory. The examples make the source and output paths explicit so existing
data can stay outside the checkout.

Keep matching page names and layouts in `dpi_72` and `dpi_144` for RULER,
LongBench, and MRCR tool evaluation. Render both resolutions using the same
source records, font, and renderer. For VTCBench, see the adapter limitation
below. The [evaluation guide](../../docs/evaluation.md) describes loading,
prompts, scoring, and custom benchmark configurations.

## A document

```bash
python eval/rendering/render_document.py \
  --text-file /path/to/document.txt \
  --out-root "$FOCUSVTC_DATA_ROOT/demo" \
  --font-path "$FOCUSVTC_FONT_PATH" \
  --dpis 72,144
```

The result contains `dpi_72/page_001.png`, `dpi_144/page_001.png`, and a
`manifest.json` with ordered image paths and page sizes. Both resolutions come
from the same PDF. Use a separate output directory for each document.

## RULER v1

Supply an existing benchmark JSONL file with these fields:

| Field | Meaning |
| --- | --- |
| `task` | RULER task name |
| `context` | Text to render into pages |
| `question` | Evaluation question |
| `answer_prefix` | Prefix used when prompting for an answer |
| `answer` | Reference answer |
| `max_new_tokens` | Answer generation budget |

```bash
python eval/rendering/render_ruler_v1.py \
  --src "$FOCUSVTC_DATA_ROOT/ruler_jsonl/ruler_8192.jsonl" \
  --out-root "$FOCUSVTC_DATA_ROOT/RULER_v1_VTC" \
  --font-path "$FOCUSVTC_FONT_PATH" \
  --dpis 72,144 --per-task 100 --processes 8
```

`--per-task 100` renders the first 100 records for each task and is the default;
use `--per-task 0` for all records. Each context is typeset once and rasterized
at every requested DPI. The renderer assigns task-local sample IDs and writes
`dpi_<dpi>/index.jsonl`, `dpi_<dpi>/images/<task>/<id>/page_001.png`, and a root
`manifest.json`. Its indexes retain questions, answers, and generation settings.

## RULER v2

Place prepared task files at `sources/ruler_v2/<task>/test.jsonl`. Records must
include `index`, `question`, `expected_answer`, and `length`. The bundled
`ruler2_split.py` separates the task instructions and question from the
haystack in the supported RULER v2 prompt formats.

```bash
python eval/rendering/render_ruler_v2.py \
  --src-root "$FOCUSVTC_DATA_ROOT/sources/ruler_v2" \
  --out-root "$FOCUSVTC_DATA_ROOT/RULER_v2_VTC" \
  --font-path "$FOCUSVTC_FONT_PATH" \
  --dpis 72 144 --processes 8
```

The renderer replaces the haystack with ordered `<image>` placeholders and
retains the reference answer in `dpi_<dpi>/<task>/test.jsonl`. Images are under
`dpi_<dpi>/images/<task>/<index>/`. Use `--tasks` to select task directories and
`--num-samples` to control the number rendered per task (default 100; 0 means
all records).

## LongBench

Place the original LongBench JSONL files at
`sources/longbench/<dataset>.jsonl`. The renderer uses the prompt templates
bundled under `eval/LongBench/config/`, renders the `context` field, and keeps
the instruction and `input` question as text. Source records also provide
`_id`, `answers`, `all_classes`, `language`, and `length`.

```bash
python eval/rendering/render_longbench.py \
  --src-root "$FOCUSVTC_DATA_ROOT/sources/longbench" \
  --out-root "$FOCUSVTC_DATA_ROOT/LongBench_VTC" \
  --font-path "$FOCUSVTC_FONT_PATH" \
  --dpis 72 144 --processes 8
```

For contexts containing CJK characters, including in otherwise English
subsets, configure a CJK font covering the input text as described above.
An optional monospace override applies to `lcc` and `repobench-p` when no CJK
font is needed. Use `--datasets` to select subsets and `--num-samples` to limit
each subset (default 0 means all records).
Use `--prompts` to supply a different prompt-template JSON file. Output JSONL files retain
answers, class labels, length metadata, and a prompt containing one `<image>`
placeholder per page. Images are under `dpi_<dpi>/images/<dataset>/<index>/`.

## MRCR

Place the original MRCR Parquet files under
`sources/mrcr/{2needle,4needle,8needle}/`. The renderer reads conversation
history from `prompt`, keeps the final user request as text, and retains
`answer` and `random_string_to_prepend`. It expects eight source blocks of
100 rows per subset, which it orders by median `n_chars` to recover the eight
length bins from `4K-8K` through `512K-1M`. Use the original source layout;
arbitrarily reshuffled or truncated Parquet files do not meet this contract.

The fast entry point reuses `render_mrcr.py` with bounded groups of text lines
for laying out long histories. Run the same entry point for both resolutions:

```bash
for dpi in 72 144; do
  python eval/rendering/render_mrcr_fast.py \
    --src-root "$FOCUSVTC_DATA_ROOT/sources/mrcr" \
    --out-root "$FOCUSVTC_DATA_ROOT/MRCR_VTC" \
    --font-path "$FOCUSVTC_FONT_PATH" \
    --dpi "$dpi" --processes 8
done
```

Use `--subsets` to select needle counts, `--num-samples` to cap each subset,
and `--recover` to reuse completed sample renders. The output includes
`length_bin`, source metadata, ordered image paths, and page sizes in
`dpi_<dpi>/<subset>/test.jsonl`; images are under
`dpi_<dpi>/images/<subset>/<index>/`.

## VTCBench-Wild

Supply the text-bearing VTCBench-Wild JSON dumps at
`sources/vtcbench/<tier>/<split>-*.json`, where the default tiers are
`data_8k`, `data_8k_16k`, `data_16k_32k`, and `data_32k_64k`, and splits are
`Retrieval`, `Reasoning`, and `Memory`. Each sample must have `_context` text.
The renderer preserves source fields, including `_context` and `_gt`, and
replaces `images` with the rendered page paths.

```bash
python eval/rendering/render_vtcbench.py \
  --source "$FOCUSVTC_DATA_ROOT/sources/vtcbench" \
  --output "$FOCUSVTC_DATA_ROOT/VTCBench-Wild_VTC" \
  --font-path "$FOCUSVTC_FONT_PATH" \
  --dpis 72 144 --processes 8
```

Use `--tiers`, `--splits`, and `--limit` to choose input coverage. JSON output
mirrors the original tier structure under each DPI directory; each split's
pages are under `<tier>/images/<split>/<index>/`.

The external VTCBench launcher defaults to the original `VTCBench-Wild`
images. Its installed version must support file paths in the `images` field
to consume these outputs. Use a custom configuration extending
`eval/tool_agent/configs/vtcbench.json` and set `data_root` to
`$FOCUSVTC_DATA_ROOT/VTCBench-Wild_VTC/dpi_72`. The current adapter crops its
source images and does not resolve paired high-resolution pages from inline
image data. Rendering `dpi_144` does not change that adapter behavior.
