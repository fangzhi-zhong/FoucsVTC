#!/usr/bin/env python3
"""Select a task-rebalanced 5K shard from the existing 50K training rows.

Use the training image-header/tokenizer length calculation, keep the original
rows, and save row indices, measured lengths and actual quotas for review.
No model weights, GPU, image copies or generated examples are needed.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sample_qwen35_vtc_grpo_train import _allocate_proportional_quotas, _write_json_atomic


GRPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/vepfs-mlp2/c20250405/400042/data/VTC/GRPO')
DEFAULT_MODEL = '/vepfs-mlp2/c20250405/400042/VTC/train/SFT/output/qwen3_5_9b_vtc_250k_random_grounding_freeze_visual/checkpoint-7000_merged'

# Explicit task quotas, rather than reproducing the QA-heavy source mix.
# LongBench "long" has only 23-page rows in this shard and does not fit 8K.
QUOTAS = {
    'gemini-3.5-flash-30k': {
        'ChatQA-Training-Data/drop': 150,
        'ChatQA-Training-Data/quoref': 125,
        'ChatQA-Training-Data/ropes': 125,
        'ChatQA-Training-Data/tatqa': 150,
        'ChatQA2-Long-SFT-data/NarrativeQA_131072': 50,
        'ChatQA2-Long-SFT-data/long_sft': 350,
        'multihop/2wiki': 75, 'multihop/2wiki_long': 150,
        'multihop/finqa': 100,
        'multihop/hotpotqa': 75, 'multihop/hotpotqa_long': 200,
        'multihop/musique': 75, 'multihop/musique_long': 150,
        'trivia_qa/rc_json': 225,
    },
    'LongBench_SFT': {'code_complete': 400, 'code_page': 250, 'count_pcnt': 600},
    'MRCR_SFT': {'mrcr_4needle': 250, 'mrcr_8needle': 250},
    'RULER_v1_SFT': {
        'count_cwe': 125, 'count_fwe': 125, 'cwe': 50, 'fwe': 50,
        'needle_multikey': 25, 'needle_single': 25,
        'niah_multikey_1': 25, 'niah_multikey_2': 25, 'niah_multikey_3': 25,
        'niah_multiquery': 50, 'niah_multivalue': 50,
        'niah_single_1': 25, 'niah_single_2': 25, 'niah_single_3': 25,
        'qa_1': 25, 'qa_2': 25, 'vt': 50,
    },
    # mv_niah_{easy,medium,hard} often ask to copy several long passages;
    # focus this short-trajectory repair set on bounded answers instead.
    'RULER_v2_SFT': {
        'mk_niah_basic': 50, 'mk_niah_easy': 50,
        'mk_niah_medium': 50, 'mk_niah_hard': 50,
        'mv_niah_basic': 100,
        'qa_basic': 50, 'qa_easy': 50, 'qa_medium': 50, 'qa_hard': 50,
    },
}


def measure_lengths(table, model, workers):
    # Reuse the exact prompt rendering and image arithmetic used at training
    # startup, without constructing a dataset or preprocessing image pixels.
    sys.path.insert(0, str(GRPO_ROOT))
    from transformers import AutoProcessor
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.dataset.prompt_length import estimate_image_tokens

    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    reader = object.__new__(RLHFDataset)
    reader.processor, reader.tokenizer = processor, processor.tokenizer
    for key, value in dict(prompt_key='prompt', image_key='images', video_key='videos',
                           tools_key='tools', tools_enabled_key='enable_tools').items():
        setattr(reader, key, value)
    reader.default_tools = json.loads(
        (GRPO_ROOT / 'examples/agent/qwen3_vl_vtc_tool/zoom_region_tools.json').read_text()
    )

    @lru_cache(maxsize=None)
    def image_tokens(path):
        return estimate_image_tokens(path, processor.image_processor)

    def measure(row):
        prompt, _ = reader._render_prompt(copy.deepcopy(row))
        prompt_tokens = len(processor.tokenizer.encode(prompt, add_special_tokens=False))
        prompt_tokens += sum(image_tokens(path) - 1 for path in row['images'])
        answer_tokens = len(processor.tokenizer.encode(
            '\n'.join(row['extra_info']['gold']), add_special_tokens=False
        ))
        return prompt_tokens, answer_tokens

    rows = table.select(['extra_info', 'prompt', 'images', 'enable_tools']).to_pylist()
    lengths = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(measure, rows), 1):
            lengths.append(result)
            if index % 5000 == 0:
                print(f'Measured {index:,} / {len(rows):,} rows', flush=True)
    return lengths


def distribution(metadata, field):
    return dict(sorted(Counter(row[field] for row in metadata).items(), key=lambda item: str(item[0])))


def select(args):
    manifest_path = args.output.with_suffix('.manifest.json')
    if args.output.resolve() == args.input.resolve():
        raise ValueError('The selected shard must have a new output path')
    for path in (args.output, manifest_path):
        if path.exists():
            raise FileExistsError(f'Output already exists: {path}')
    table = pq.read_table(args.input)
    metadata = table['extra_info'].to_pylist()
    if args.length_profile:
        profile = json.loads(args.length_profile.read_text())
        if profile['model'] != str(args.model) or profile['input'] != str(args.input.resolve()):
            raise ValueError('Length profile belongs to a different input or model')
        if profile['row_ids'] != [[row['source'], row['id'], row['dpi']] for row in metadata]:
            raise ValueError('Length profile row order does not match the source shard')
        lengths = profile['lengths']
    else:
        lengths = measure_lengths(table, str(args.model), args.workers)
    if len(lengths) != table.num_rows:
        raise ValueError('Length profile does not match the input row count')

    # Import the reward's evidence normalization so the manifest uses exactly
    # the same valid, unique (page, bbox) count as training.
    sys.path.insert(0, str(GRPO_ROOT / 'examples/reward_function'))
    from qwen35_vtc_reward import _evidence_pairs

    evidence_counts = [len(_evidence_pairs(row)) for row in metadata]
    eligible = defaultdict(list)
    exclusions = Counter()
    for i, (prompt_tokens, answer_tokens) in enumerate(lengths):
        reason = ('prompt_over_limit' if prompt_tokens > args.max_prompt_tokens else
                  'answer_over_limit' if answer_tokens > args.max_answer_tokens else
                  'empty_answer' if answer_tokens == 0 else None)
        if reason:
            exclusions[reason] += 1
            continue
        row = metadata[i]
        eligible[(row['source'], row['subset'])].append(i)

    rng = random.Random(args.seed)
    selected, quota_report = [], []
    seen_ids = set()
    for source, subsets in QUOTAS.items():
        for subset, quota in subsets.items():
            candidates = list(eligible[(source, subset)])
            rng.shuffle(candidates)
            # Keep one DPI view per exact source/id, maximizing problem
            # diversity. Source IDs containing DPI are not fuzzy-merged.
            unique, local_seen = [], set()
            for i in candidates:
                identity = (source, metadata[i]['id'])
                if identity not in local_seen and identity not in seen_ids:
                    unique.append(i)
                    local_seen.add(identity)
            if len(unique) < quota:
                raise ValueError(f'{source}/{subset}: need {quota}, only {len(unique)} eligible unique IDs')
            strata = defaultdict(list)
            for i in unique:
                # Preserve DPI and actual short/long-within-8K variation,
                # plus zero/single/multiple-evidence examples within tasks.
                key = (metadata[i]['dpi'], 'le4k' if lengths[i][0] <= 4096 else '4k_8k',
                       min(evidence_counts[i], 3))
                strata[key].append(i)
            allocations = _allocate_proportional_quotas({k: len(v) for k, v in strata.items()}, quota)
            chosen = []
            for key in sorted(strata):
                rng.shuffle(strata[key])
                chosen.extend(strata[key][:allocations[key]])
            selected.extend(chosen)
            seen_ids.update((source, metadata[i]['id']) for i in chosen)
            quota_report.append({'source': source, 'subset': subset, 'count': quota,
                                 'eligible_rows': len(candidates), 'eligible_unique_ids': len(unique)})
    rng.shuffle(selected)
    if len(selected) != 5000 or len(set(selected)) != 5000:
        raise RuntimeError('The fixed task quotas must select exactly 5,000 distinct rows')

    selected_table = table.take(pa.array(selected, type=pa.int64()))
    selected_meta = [metadata[i] for i in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name('.' + args.output.name + '.tmp')
    pq.write_table(selected_table, temporary, compression='snappy')
    temporary.replace(args.output)
    manifest = {
        'input': str(args.input.resolve()), 'output': str(args.output.resolve()),
        'input_rows': table.num_rows, 'output_rows': len(selected), 'seed': args.seed,
        'model_for_length_measurement': str(args.model),
        'length_method': 'training chat template + tokenizer + image header merged visual tokens',
        'max_prompt_tokens': args.max_prompt_tokens, 'max_answer_tokens': args.max_answer_tokens,
        'sampling_method': 'fixed task quotas; proportional DPI x actual length x evidence strata; one exact source/id',
        'exclusions_in_priority_order': dict(exclusions), 'quotas': quota_report,
        'source_distribution': distribution(selected_meta, 'source'),
        'dpi_distribution': distribution(selected_meta, 'dpi'),
        'page_distribution': distribution(selected_meta, 'num_pages'),
        'evidence_count_distribution': dict(sorted(Counter(evidence_counts[i] for i in selected).items())),
        'actual_prompt_length_buckets': {'le4k': sum(lengths[i][0] <= 4096 for i in selected),
                                         '4k_8k': sum(lengths[i][0] > 4096 for i in selected)},
        'prompt_tokens_max': max(lengths[i][0] for i in selected),
        'answer_tokens_max': max(lengths[i][1] for i in selected),
        'selected_row_indices': selected,
        'selected_prompt_tokens': [lengths[i][0] for i in selected],
        'selected_answer_tokens': [lengths[i][1] for i in selected],
        'limitations': [
            'No new summary/few-shot tasks can be created by resampling existing rows.',
            '8K selection cannot cover the evaluation distribution of very long documents.',
            'Evidence count means unique annotated regions, not semantic facts; the 50K shard has no evidence IDs.',
            'Image paths, prompts, answers, enable_tools and max_tool_calls remain the original row values.',
        ],
    }
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({key: value for key, value in manifest.items() if not key.startswith('selected_')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_ROOT / 'train.parquet')
    parser.add_argument('--output', type=Path, default=DATA_ROOT / 'train_5k_tool_rebalanced.parquet')
    parser.add_argument('--model', type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument('--max-prompt-tokens', type=int, default=8192)
    parser.add_argument('--max-answer-tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--length-profile', type=Path, help='optional profile from the same unchanged input and model')
    select(parser.parse_args())
