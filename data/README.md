# REL-CoT rendering and SFT preparation

Download the public [REL-CoT dataset on ModelScope](https://www.modelscope.cn/datasets/zhongfangzhi/REL-CoT)
separately. This directory contains tools and instructions only. Keep the
source data under `datasets/REL-CoT/` and prepared training data under
`datasets/REL-CoT_SFT/`, or use directories outside the checkout. Benchmark
renderers are under [`eval/rendering/`](../eval/rendering/README.md).

The supported release contains original text, 72 DPI page images, and existing
REL-CoT supervision. `render_relcot_sft.py` renders the source text at
**48, 60, 72, 84, 96, 120, and 144 DPI**, using **DejaVu Sans for every output**,
and writes SFT manifests. All seven views, including 72 DPI, are rendered
again. The tool reuses the supplied reasoning and answers without model calls
or synthetic task generation.

## Download and unpack

Run commands from the FocusVTC repository root. Install Python 3.12, the
system Poppler tools, `zstd`, and DejaVu Sans. On Debian/Ubuntu, the relevant
system packages are `poppler-utils`, `zstd`, and `fonts-dejavu-core`.
Font binaries are not bundled.

```bash
python -m pip install -r data/requirements.txt
python -m pip install modelscope
modelscope download --dataset zhongfangzhi/REL-CoT \
  --local_dir datasets/REL-CoT

cat datasets/REL-CoT/archives/images-72dpi.tar.zst.part-* \
  | tar --use-compress-program=unzstd -xf - -C datasets/REL-CoT
cat datasets/REL-CoT/archives/text.tar.zst.part-* \
  | tar --use-compress-program=unzstd -xf - -C datasets/REL-CoT
cat datasets/REL-CoT/archives/conversations.tar.zst.part-* \
  | tar --use-compress-program=unzstd -xf - -C datasets/REL-CoT
```

The expected unpacked layout is:

```text
REL-CoT/
├── train_72dpi.json
├── images/                 # Original 72 DPI reference pages
├── text/                   # Original document text
└── conversations/          # Existing reasoning, evidence, and answers
```

Despite its `.json` extension, `train_72dpi.json` is JSON Lines: one record per
line. The renderer uses it by default, with `train.jsonl` as a fallback;
`--input` selects another manifest explicitly. Each record supplies `id`, an
ordered `image_dpi72` or `image` list, `text_path`, and `conversations`.
Conversation paths and inline conversation lists are both supported.

The published manifest retains its original absolute asset prefix. The tool
maps this prefix to `--dataset-root` automatically, without modifying the
source manifest. Use `--source-prefix` if your source manifest was produced
under a different prefix.

## Render seven DPI views

```bash
python data/render_relcot_sft.py \
  --dataset-root "$PWD/datasets/REL-CoT" \
  --out-root "$PWD/datasets/REL-CoT_SFT" \
  --font-path /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
  --dpis 48 60 72 84 96 120 144 \
  --processes 8
```

The seven DPI values above are the defaults. `--font-path` defaults to
`FOCUSVTC_FONT_PATH` when set, otherwise the system DejaVu Sans path shown
above; the output font must be DejaVu Sans. `--limit N` selects the first N
source records; the default `0` processes the entire manifest. Use
`--poppler-path` if `pdftoppm` and `pdfinfo` are installed outside your PATH.

The layout uses a 9 pt font, 10 pt line height, a 595 × 842 pt page, and
10 pt margins. The original 72 DPI images are required to locate the text
covered by the existing evidence boxes. The renderer maps those characters
into the new DejaVu Sans layout, updates evidence page numbers and normalized
boxes, and adjusts page references and `<image>` placeholders in the supplied
conversation. Pixel boxes (`bbox_abs`) are calculated for each output DPI.
The reasoning and answers are retained while their page/coordinate references
are updated.

Some existing local 72 DPI pages use Verdana. For those pages, add
`--source-font-path /path/to/Verdana.ttf` to the command, pointing to your
locally available original font. Repeat this option for other original fonts
when needed. These fonts are used only to recover the existing evidence
locations; every output still uses DejaVu Sans. No extra source font is
needed when the original pages already use DejaVu Sans. A sample that cannot
be mapped is reported in `failed.jsonl` instead of retaining evidence boxes
from an incompatible layout.

To continue an interrupted run, repeat the command with `--resume`. Completed
samples with unchanged inputs are reused. Keep the same rendering options;
use a fresh output directory when changing fonts or DPI selection. Final
manifests are published after all selected samples complete successfully.

## Outputs

```text
REL-CoT_SFT/
├── train.jsonl             # All selected DPI variants
├── train_48dpi.jsonl        # One manifest per DPI
├── train_60dpi.jsonl
├── train_72dpi.jsonl
├── train_84dpi.jsonl
├── train_96dpi.jsonl
├── train_120dpi.jsonl
├── train_144dpi.jsonl
├── images/dpi_<dpi>/sample_<index>/
├── conversations/          # Conversations with remapped page/box references
├── state/                  # Per-source records for --resume
├── render_config.json
└── failed.jsonl
```

A successful default run over N source samples produces 7N rows in
`train.jsonl`. Each row has an ID of `<original-id>__<dpi>dpi` and retains the
source ID in `metadata.base_id`, with the view DPI in `metadata.dpi`.
The row's `image`, page sizes, evidence locations, and conversation path refer
to the new rendering. Old token/patch metadata and unavailable image paths
are removed. When 144 DPI is selected, `high_res_images` points to the
rendered 144 DPI pages for subsequent GRPO conversion.
GRPO additionally requires independent reference answers in `metadata.gold`
or `gold`; the released REL-CoT manifest does not provide that field. The
renderer does not derive it from the supervised assistant response.

The manifests use absolute asset paths. Original text paths still point into
the downloaded dataset, so retain that directory as well as the output
images and conversations. Do not move either tree without updating its
manifest paths.

## Use for training

```bash
export FOCUSVTC_SFT_DATASET="$PWD/datasets/REL-CoT_SFT/train.jsonl"
export FOCUSVTC_SFT_ASSETS="$PWD/datasets/REL-CoT_SFT"
NGPUS=8 bash train/SFT/run.sh
```

Install the SFT environment and select a base model as described in the
[training guide](../docs/training.md). A per-DPI manifest can be used in place
of the combined `train.jsonl`.

Split training and validation by **`metadata.base_id`**, keeping all DPI
variants of a source document in the same split. Splitting individual rows
can put different renderings of the same document in both sets. The
[GRPO data instructions](../docs/training.md#grpo-data) describe converting
manifests with reference answers and paired low- and high-resolution pages
into Parquet.
