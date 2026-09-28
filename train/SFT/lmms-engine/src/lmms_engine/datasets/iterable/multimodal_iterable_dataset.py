import math
import os
import random
import traceback
from copy import deepcopy
import bisect

import torch.distributed as dist
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from loguru import logger
from torch.utils.data import get_worker_info

from lmms_engine.datasets.multimodal_mixin import MultiModalDataLoadingMixin
from lmms_engine.utils import DataUtilities

try:
    import lmms_engine.parallel.process_group_manager as pgm
except ImportError:
    pgm = None

try:
    from google.cloud.storage import Client
except ImportError:
    logger.info("Google Cloud SDK not installed. Skipping import.")

try:
    from azure.storage.blob import BlobServiceClient, LinearRetry

    RETRY_POLICY = LinearRetry(backoff=10, retry_total=5, random_jitter_range=0)
    SAS_URL = os.environ.get("AZURE_STORAGE_SAS_URL", "YOUR_SAS_URL")
except ImportError:
    logger.info("Azure SDK not installed. Skipping import.")

from lmms_engine.datasets.iterable.base_iterable_dataset import BaseIterableDataset


class MultiModalIterableDataset(BaseIterableDataset, MultiModalDataLoadingMixin):
    """
    MultiModalDataset provides concrete implementation for handling multimodal data
    including images, audio, and videos with support for various data formats and
    object storage backends.

    This class inherits from BaseDataset and implements all the abstract methods
    with full functionality for data loading, processing, and packing.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        # Initialize object storage clients if needed
        if self.config.object_storage == "gcs":
            self.storage_client = Client()
            self.bucket_name = self.config.bucket_name
        elif self.config.object_storage == "azure":
            self.storage_client = BlobServiceClient(account_url=SAS_URL, retry_policy=RETRY_POLICY)
            self.bucket_name = self.config.bucket_name
        self.cur_idx = 0
        if not dist.is_initialized():
            logger.info("Distributed environment not initialized, setting rank and world size to 0 and 1")
            self.rank = 0
            self.world_size = 1
        else:
            logger.info(
                "Distributed environment initialized, setting rank and world size to dist.get_rank() and dist.get_world_size()"
            )
            # Try to use data parallel rank if available, otherwise fall back to global rank
            if pgm is not None and hasattr(pgm, "process_group_manager") and pgm.process_group_manager is not None:
                self.rank = pgm.process_group_manager.dp_rank
                self.world_size = pgm.process_group_manager.dp_world_size
            else:
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()

    def _build_from_config(self):
        """Load and prepare data from the configuration."""
        if self.config.dataset_format == "json":
            self.data_list = DataUtilities.load_json(self.config.dataset_path)
        elif self.config.dataset_format == "jsonl":
            self.data_list = DataUtilities.load_jsonlines(self.config.dataset_path)
        elif self.config.dataset_format == "arrow":
            self.data_list = load_from_disk(self.config.dataset_path)
        elif self.config.dataset_format == "parquet":
            self.data_list = HFDataset.from_parquet(self.config.dataset_path)
        elif self.config.dataset_format == "hf_dataset":
            self.data_list = load_dataset(self.config.dataset_path, split="train")
            self.data_list_no_image = deepcopy(self.data_list)
            self.data_list_no_image = self.data_list_no_image.remove_columns("image")
        elif self.config.dataset_format == "yaml":
            # Handle both external YAML files and inline datasets
            if self.config.datasets is not None:
                # Use inline datasets defined in the config
                self.data_list, self.data_folder = DataUtilities.load_inline_datasets(self.config.datasets)
            elif self.config.dataset_path is not None:
                # Load from external YAML file
                self.data_list, self.data_folder = DataUtilities.load_yaml(self.config.dataset_path)
            else:
                raise ValueError("For yaml format, either 'datasets' or 'dataset_path' must be provided")
        else:
            raise NotImplementedError

        if self.config.shuffle:
            logger.info("Shuffle Dataset ...")
            data_index = [i for i in range(len(self.data_list))]
            # Make sure the shuffle is the same across all dp ranks
            random.seed(self.config.data_seed)
            random.shuffle(data_index)
            if isinstance(self.data_list, HFDataset):
                self.data_list = self.data_list.select(data_index)
            else:
                self.data_list = [self.data_list[i] for i in data_index]
            if getattr(self, "data_folder", None) is not None:
                self.data_folder = [self.data_folder[i] for i in data_index]

    def get_one_sample(self, index, data_folder=None, data_list=None):
        """Get a sample from the dataset by index."""
        if data_folder is None:
            data_folder = self.data_folder[index]
        if data_list is None:
            data_list = self.data_list
        if (
            self.config.dataset_format == "json"
            or self.config.dataset_format == "jsonl"
            or self.config.dataset_format == "arrow"
        ):
            data_dict = self.load_from_json(data_list[index])
        elif self.config.dataset_format == "yaml":
            data_dict = self.load_from_json(data_list[index], data_folder)
        elif self.config.dataset_format == "hf_dataset":
            data_dict = self.load_from_hf(data_list[index])
        else:
            raise NotImplementedError
        return data_dict

    def __iter__(self):
        worker_info = get_worker_info()
        rank = self.rank
        world_size = self.world_size

        assert isinstance(self.data_list, HFDataset), "Data list must be a HuggingFace dataset for IterableDataset"

        # HF shard logic, if len(dataset) % n == l
        # The first l ranks will have dataset length (len(dataset) // n) + 1
        # The rest ranks will have dataset length (len(dataset) // n)
        rank_mod_size = len(self.data_list) % world_size
        per_rank_size = [
            (len(self.data_list) // world_size) + 1 if i < rank_mod_size else (len(self.data_list) // world_size)
            for i in range(world_size)
        ]
        start_index = sum(per_rank_size[:rank])
        end_index = start_index + per_rank_size[rank]

        # Shard the data according to distributed environment
        curr_data_folder = self.data_folder[start_index:end_index]
        # self.data_folder = self.data_folder[start_index:end_index]
        curr_data_list = self.data_list.shard(num_shards=world_size, index=rank, contiguous=True)

        if worker_info is None:
            iter_start = 0
            iter_end = len(curr_data_list)
        else:
            # split workload
            per_worker = int(math.ceil(len(curr_data_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(curr_data_list))

        # Distrbute the data to each worker
        curr_data_list = curr_data_list.select(range(iter_start, iter_end))
        if getattr(self, "data_folder", None) is not None:
            curr_data_folder = curr_data_folder[iter_start:iter_end]

        if self.config.packing:

            # k_bucket_batching is the pre-refactor name for length_grouped.
            if self.config.packing_strategy in ('length_grouped', 'k_bucket_batching'):
                yield from self._iter_packed_length_grouped(
                    curr_data_list=curr_data_list, curr_data_folder=curr_data_folder
                )
            elif self.config.packing_strategy=='best_fit_lookahead':
                yield from self._iter_packed_best_fit_lookahead(curr_data_list=curr_data_list, curr_data_folder=curr_data_folder)
            else:
                yield from self._iter_packed_first_fit(curr_data_list=curr_data_list, curr_data_folder=curr_data_folder)
            
        else:
            self.cur_idx = 0
            while self.cur_idx < len(curr_data_list):
                try:
                    yield self.get_one_sample(self.cur_idx, curr_data_folder[self.cur_idx], curr_data_list)
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error getting one sample: {e}, skip this sample")
                    self.cur_idx += 1
                    continue
                self.cur_idx += 1

    def _iter_packed_first_fit(self, curr_data_list, curr_data_folder):
        self.cur_idx = 0
        buffer = []
        buffer_length = 0
        packing_length = self.config.packing_length

        # Iterate through the dataset once per epoch
        while self.cur_idx < len(curr_data_list):
            try:
                data_dict = self.get_one_sample(self.cur_idx, curr_data_folder[self.cur_idx], curr_data_list)
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error getting one sample: {e}, skip this sample")
                self.cur_idx += 1
                continue
            input_ids = data_dict["input_ids"]
            data_length = input_ids.shape[0]
            self.cur_idx += 1

            if data_length > self.config.max_length:
                continue
            
            # Drop overlong sample if filtering is enabled
            if data_length > packing_length and self.config.filter_overlong:
                continue

            # If current sample cannot fit into current buffer, yield the buffer first
            if buffer_length > 0 and buffer_length + data_length > packing_length:
                yield buffer
                buffer = []
                buffer_length = 0

            # If the sample is still longer than packing_length (and not filtered),
            # yield it as its own batch to avoid stalling
            if data_length > packing_length:
                yield [data_dict]
                continue

            # Append to buffer
            buffer.append(data_dict)
            buffer_length += data_length

        # Flush remaining buffer
        if len(buffer) > 0:
            yield buffer

    
    # A pool bounded only by tokens can still hold a huge number of objects when
    # the corpus has a short-sample mode, which makes the per-drain sort and the
    # collator's Python loop the bottleneck. Cap the count as well.
    MAX_POOL_ITEMS = 1024

    def _iter_packed_length_grouped(self, curr_data_list, curr_data_folder):
        """Length-grouped packing over a bounded look-ahead pool.

        Samples accumulate in one pool until it holds ``packing_pool_tokens``,
        then the *whole* pool is sorted by length and cut into packs. Because
        every drain empties the pool, no length range can quietly accumulate:
        each sample leaves within one pool-fill, which bounds both buffered
        memory and how stale a sample can get.

        This replaces a fixed-bucket scheme that only flushed a bucket once it
        could fill a pack on its own. A bucket of very short samples needs
        ``packing_length / bucket_max_len`` items to reach that point -- around
        100 for the shortest bucket -- so with the shard sizes an 8-rank x
        8-worker dataloader produces (a few hundred samples each), most buckets
        never filled and roughly a third of all packs came from the
        end-of-shard flush as thin, half-empty batches.

        Two budgets are supported, and the right one depends on the model:

        ``packing_cost="padded"``
            ``num_items * max_len <= packing_length``. The collator pads every
            sample to the batch max, so this bounds the tensor the model
            actually sees. Required when the model consumes that padded tensor,
            i.e. ``use_rmpad`` is off.
        ``packing_cost="tokens"``
            ``sum(len) <= packing_length``. Correct when ``use_rmpad`` is on,
            since the model strips padding before doing any work and cost then
            scales with real tokens. Fits noticeably more data per step, because
            the padded budget has to reserve room for padding that gets thrown
            away.
        """
        self.cur_idx = 0
        packing_length = self.config.packing_length
        max_length = self.config.max_length or packing_length
        cost = getattr(self.config, "packing_cost", "padded")
        pool_token_limit = getattr(self.config, "packing_pool_tokens", None) or 8 * packing_length
        max_items_per_pack = self.config.packing_max_items_per_pack or 0
        # Under the token budget a single long sample will happily absorb a tail
        # of very short ones; the collator still materialises [B, max_len], so
        # bound that shape separately or it reaches many times packing_length.
        padded_limit = 4 * packing_length

        pool = []  # list[(length, data_dict)], unsorted until a drain
        pool_tokens = 0

        def drain(final):
            """Yield every pack of one drain, dropping each one as it goes out.

            Holding the whole list would keep a pool's worth of decoded samples
            alive until the last of them had been consumed.
            """
            packs = build_packs(final)
            while packs:
                chunk = packs.pop()
                yield [data for _, data in chunk]

        def build_packs(final):
            """Sort the pool by length and cut it into packs, emptying the pool."""
            nonlocal pool, pool_tokens
            pool.sort(key=lambda item: -item[0])
            packs, carry, i = [], [], 0
            while i < len(pool):
                head = pool[i][0]  # sorted descending, so this is the pack's max_len
                chunk, used = [pool[i]], head
                i += 1
                while i < len(pool):
                    if max_items_per_pack and len(chunk) >= max_items_per_pack:
                        break
                    nxt = pool[i][0]
                    if cost == "padded":
                        if (len(chunk) + 1) * head > packing_length:
                            break
                    else:
                        if used + nxt > packing_length:
                            break
                        if (len(chunk) + 1) * head > padded_limit:
                            break
                    chunk.append(pool[i])
                    used += nxt
                    i += 1
                # Only the final chunk can be underfull. Hold it back so it
                # merges with the next refill rather than going out half empty.
                if not final and i >= len(pool) and used * 2 < packing_length:
                    carry = chunk
                else:
                    packs.append(chunk)
            pool = carry
            pool_tokens = sum(length for length, _ in pool)
            # Without this the packs of one drain come out longest-first, which
            # turns per-step sequence length into a sawtooth.
            random.shuffle(packs)
            return packs

        while self.cur_idx < len(curr_data_list):
            try:
                data_dict = self.get_one_sample(self.cur_idx, curr_data_folder[self.cur_idx], curr_data_list)
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error getting one sample at idx={self.cur_idx}: {e}, skip this sample")
                self.cur_idx += 1
                continue

            input_ids = data_dict.get("input_ids")
            if input_ids is None:
                self.cur_idx += 1
                continue
            data_length = int(input_ids.shape[0])
            self.cur_idx += 1

            if data_length > max_length:
                continue
            if data_length > packing_length:
                if self.config.filter_overlong:
                    continue
                yield [data_dict]
                continue

            pool.append((data_length, data_dict))
            pool_tokens += data_length

            if pool_tokens >= pool_token_limit or len(pool) >= self.MAX_POOL_ITEMS:
                yield from drain(final=False)

        yield from drain(final=True)

    def _iter_packed_best_fit_lookahead(self, curr_data_list, curr_data_folder):
        """
        Lookahead + Best-Fit packing:
        - 预读一个窗口到 pool（按 length 排序）
        - 每次从 pool 里取 <= remaining 的最大样本填充
        - pool 不够时继续预读
        目标：每个 pack 尽量接近 packing_length
        """
        self.cur_idx = 0
        packing_length = self.config.packing_length
        max_length = self.config.max_length
        filter_overlong = self.config.filter_overlong

        # 你可以在 config 里加这些参数
        lookahead = getattr(self.config, "packing_lookahead", None)   # 候选池最大容量（越大越满，但更耗时/内存）
        max_items_per_pack = getattr(self.config, "packing_max_items_per_pack", None)  # 防极端短样本

        if lookahead is None:
            lookahead = 256  # 默认值
        if max_items_per_pack is None:
            max_items_per_pack = 1e5  # 不限制
        # pool 维护两个并行结构，支持二分：
        # pool_lens: [len1, len2, ...] 升序
        # pool_data: [(len, tie_id, data_dict), ...] 升序，tie_id 用来稳定排序
        pool_lens = []
        pool_data = []
        tie_id = 0  # 保证稳定、可复现

        def push_pool(data_dict, data_length):
            nonlocal tie_id
            # 按 (len, tie_id) 排序插入
            pos = bisect.bisect_right(pool_lens, data_length)
            pool_lens.insert(pos, data_length)
            pool_data.insert(pos, (data_length, tie_id, data_dict))
            tie_id += 1

        def pop_best_leq(remaining):
            """弹出 pool 中 <= remaining 的最大元素；若不存在返回 None"""
            idx = bisect.bisect_right(pool_lens, remaining) - 1
            if idx < 0:
                return None
            pool_lens.pop(idx)
            return pool_data.pop(idx)[2]  # data_dict

        def refill_pool(target_size):
            """把 pool 补到 target_size（或数据耗尽）"""
            nonlocal max_length, filter_overlong, packing_length
            while len(pool_data) < target_size and self.cur_idx < len(curr_data_list):
                try:
                    data_dict = self.get_one_sample(
                        self.cur_idx, curr_data_folder[self.cur_idx], curr_data_list
                    )
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error getting one sample: {e}, skip this sample")
                    self.cur_idx += 1
                    continue

                input_ids = data_dict["input_ids"]
                data_length = int(input_ids.shape[0])
                self.cur_idx += 1

                # 过滤 max_length
                if data_length > max_length:
                    continue

                # 过滤 packing_length
                if data_length > packing_length and filter_overlong:
                    continue

                # 超过 packing_length 但不过滤：直接单独 yield（避免卡住）
                if data_length > packing_length:
                    yield [data_dict]
                    continue

                push_pool(data_dict, data_length)

        # 主循环：不断生成 pack
        while True:
            # 先把 pool 填起来
            # refill_pool 里可能会 yield（单独 overlong 样本），所以用生成器接住
            yielded_any = False
            for maybe_single in refill_pool(lookahead):
                yielded_any = True
                yield maybe_single

            # 数据耗尽且 pool 为空：结束
            if self.cur_idx >= len(curr_data_list) and len(pool_data) == 0:
                break

            # 开始装一个 pack
            pack = []
            used = 0
            items = 0

            while items < max_items_per_pack:
                remaining = packing_length - used
                if remaining <= 0:
                    break

                best = pop_best_leq(remaining)
                if best is None:
                    # pool 里找不到能塞进 remaining 的样本：
                    # 尝试再读一些数据补 pool（可能会带来更短样本）
                    for maybe_single in refill_pool(lookahead):
                        yielded_any = True
                        yield maybe_single

                    best = pop_best_leq(remaining)
                    if best is None:
                        break  # 还是没有 -> 结束这个 pack

                pack.append(best)
                used += int(best["input_ids"].shape[0])
                items += 1

                # 可选：如果 pool 太空，继续补一点，帮助下一次更满
                if len(pool_data) < lookahead // 2 and self.cur_idx < len(curr_data_list):
                    for maybe_single in refill_pool(lookahead):
                        yielded_any = True
                        yield maybe_single

            # pack 是否产出：按最小填充率控制（默认直接产出）
            if len(pack) > 0:
                print(f'GPU:{self.rank} Yield pack of size {len(pack)}, total length {used}')
                yield pack


    def load_from_json(self, data, data_folder=None):
        """
        Default implementation for loading from JSON format.
        Subclasses should override this method to provide specific implementations.

        Args:
            data: The JSON data to process
            data_folder: Optional folder path for data files

        Returns:
            Processed data dictionary
        """
        raise NotImplementedError("Subclasses must implement load_from_json")

    def load_from_hf(self, data):
        """
        Default implementation for loading from HuggingFace dataset format.
        Subclasses should override this method to provide specific implementations.

        Args:
            data: The HuggingFace dataset data to process

        Returns:
            Processed data dictionary
        """
        raise NotImplementedError("Subclasses must implement load_from_hf")

    def get_collator(self):
        """
        Get the appropriate collator for this dataset.
        Subclasses should override this method to provide specific implementations.

        Returns:
            A collator instance suitable for this dataset type
        """
        raise NotImplementedError("Subclasses must implement get_collator")
