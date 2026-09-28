# Font and point-size calibration

These utilities render A4 pages at 72 DPI, ask a vision model to transcribe
them, and measure character error rate (CER). Synthetic random strings and
confusable character groups help separate character recognition from language
priors. `acuity_lib.py` contains threshold interpolation and bootstrap helpers.
No calibration measurements, source corpora, or font binaries are included.

Run from the repository root. Install Poppler (including `pdftoppm`) through
your system package manager, then install the Python dependencies:

```bash
python -m pip install -r eval/font_acuity/requirements.txt
export FOCUSVTC_FONT_PATH=/path/to/DejaVuSans.ttf
python eval/font_acuity/render.py --fonts dejavu \
  --conds random confusable --sizes 6 7 8 9 10 11 \
  --out outputs/font_acuity
```

For multiple fonts, pass `--font-config /path/to/fonts.json` containing a
mapping such as `{"dejavu": "/path/to/DejaVuSans.ttf", "custom": "/path/to/font.ttf"}`,
then select their keys with `--fonts dejavu custom`. Relative font paths are
resolved against the JSON file. Supply fonts under their own licenses.
An optional `--conds natural --natural-source /path/to/corpus.jsonl` reads
user-provided JSONL rows with a `context` string. Rendering overwrites pages
in the selected output directory; use separate directories for different
corpora or font configurations.

With a vision-model endpoint already running and able to read the rendered
files, run:

```bash
python eval/font_acuity/run_eval.py \
  --root outputs/font_acuity \
  --base-url http://127.0.0.1:18450/v1 --model FocusVTC \
  --out outputs/font_acuity/transcribe.jsonl
```

Use the model backend directly for this transcription task. The FocusVTC
tool gateway adds a reasoning/tool-use prompt intended for document QA.
For authenticated endpoints, set `FOCUSVTC_API_KEY`. Use a distinct output
file for each model and decoding setup; completed successful rows are skipped
when the same run is resumed.
