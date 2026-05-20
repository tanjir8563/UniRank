# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2025. FuxiCTR Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import glob
import json
import random
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler


def _resolve_parquet_files(data_path):
    """
    支持以下形式:
      - 单文件: xxx.parquet
      - 不带后缀: xxx
      - 通配符: xxx/*.parquet
      - 文件夹: xxx/
      - list/tuple
    """
    if isinstance(data_path, (list, tuple)):
        files = []
        for p in data_path:
            files.extend(_resolve_parquet_files(p))
        files = sorted(files)
        if len(files) == 0:
            raise FileNotFoundError(f"No parquet files found in: {data_path}")
        return files

    data_path = str(data_path)

    if any(ch in data_path for ch in ["*", "?", "["]):
        files = sorted(glob.glob(data_path))
        if len(files) == 0:
            raise FileNotFoundError(f"No parquet files matched: {data_path}")
        return files

    path = Path(data_path)

    if path.is_dir():
        files = sorted(str(p) for p in path.glob("*.parquet"))
        if len(files) == 0:
            raise FileNotFoundError(f"No parquet files found in directory: {data_path}")
        return files

    if not data_path.endswith(".parquet"):
        alt = data_path + ".parquet"
        if Path(alt).exists():
            data_path = alt

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Parquet path not found: {data_path}")

    return [data_path]


def _get_parquet_schema_names(files):
    """
    从第一个 parquet 文件推断 schema 列名。
    默认各分片 schema 一致。
    """
    if isinstance(files, str):
        files = _resolve_parquet_files(files)
    if len(files) == 0:
        raise FileNotFoundError("No parquet files found for schema inference.")
    return set(pq.ParquetFile(files[0]).schema.names)


def _is_sequence_like(x):
    return isinstance(x, (list, tuple, np.ndarray))


def _is_sequence_column(series):
    """
    判断一列是否为 sequence 列。
    """
    if len(series) == 0:
        return False
    for x in series:
        if x is None:
            continue
        return _is_sequence_like(x)
    return False


def _dataframe_to_darray(df):
    """
    把 DataFrame 转成与原实现兼容的 2D numpy array，
    并返回 column_index:
      - 标量列 -> int
      - sequence列 -> [int, int, ...]
    """
    column_index = {}
    data_arrays = []
    idx = 0

    for col in df.columns:
        series = df[col]

        if _is_sequence_column(series):
            array = np.array(series.to_list())
            if array.ndim == 1:
                array = series.to_numpy()
                column_index[col] = idx
                idx += 1
            else:
                seq_len = array.shape[1]
                column_index[col] = [idx + i for i in range(seq_len)]
                idx += seq_len
            data_arrays.append(array)
        else:
            array = series.to_numpy()
            column_index[col] = idx
            idx += 1
            data_arrays.append(array)

    if len(data_arrays) == 0:
        raise ValueError("No columns were loaded from parquet file.")

    darray = np.column_stack(data_arrays)
    return darray, column_index


def _extract_part_id(fp):
    """
    从 part-00012.parquet 提取 12
    """
    name = Path(fp).name
    m = re.match(r"part-(\d+)\.parquet$", name)
    if m is None:
        raise ValueError(f"Invalid blocked parquet filename: {fp}")
    return int(m.group(1))


def _build_part_file_map(path_like):
    """
    把一个目录 / glob / 单文件 解析成:
        {part_id: filepath}
    """
    files = _resolve_parquet_files(path_like)
    mp = {}
    for fp in files:
        pid = _extract_part_id(fp)
        if pid in mp:
            raise ValueError(f"Duplicate part id found: part-{pid:05d}")
        mp[pid] = fp
    return mp


def _find_meta_data_json(path_like):
    """
    兼容以下情况:
      - old: user_info.parquet 同目录下有 meta_data.json
      - blocked: split/user_info/ 目录，meta_data.json 在 dataset root 或更上层
      - 更深层目录时，向上逐级搜索
    """
    p = Path(str(path_like)).resolve()

    candidates = []
    cur = p if p.is_dir() else p.parent
    for _ in range(8):
        candidates.append(cur / "meta_data.json")
        if cur.parent == cur:
            break
        cur = cur.parent

    for fp in candidates:
        if fp.exists() and fp.is_file():
            return fp

    raise FileNotFoundError(
        f"meta_data.json not found from path: {path_like}\n"
        f"Tried: {[str(x) for x in candidates]}"
    )


def _build_batch_dict_from_tensor(batch_tensor, batch_cols):
    batch_dict = {}
    for col, idx in batch_cols:
        if isinstance(idx, list):
            batch_dict[col] = batch_tensor[:, idx]
        else:
            batch_dict[col] = batch_tensor[:, idx]
    return batch_dict


def _resolve_side_info_path(split, key, explicit_path=None, kwargs=None):
    """
    优先级:
      1) 显式传入 explicit_path
      2) split_key, 例如 train_user_info / valid_item_info / test_user_info
      3) 通用 key, 例如 user_info / item_info
    """
    if explicit_path is not None:
        return explicit_path
    kwargs = kwargs or {}
    if split is not None:
        split_key = f"{split}_{key}"
        if kwargs.get(split_key) is not None:
            return kwargs[split_key]
    if kwargs.get(key) is not None:
        return kwargs[key]
    raise ValueError(
        f"Missing side-info path for key='{key}', split='{split}'. "
        f"Expected one of: explicit `{key}`, `{split}_{key}`, or `{key}` in kwargs."
    )

def _estimate_block_cost(data_file, seq_len_col="seq_len", sample_rows=4096):
    """
    估计一个 blocked parquet 的训练负载。
    默认使用:
        cost = num_rows * (avg_seq_len + 1)

    说明:
    - 只采样前若干行估计 avg_seq_len，避免全量读取 parquet
    - 若不存在 seq_len 列，则退化为仅按 num_rows 估计
    """
    pf = pq.ParquetFile(data_file)
    num_rows = int(pf.metadata.num_rows)

    schema_names = set(pf.schema.names)
    if seq_len_col not in schema_names:
        return {
            "num_rows": num_rows,
            "avg_seq_len": None,
            "cost": float(num_rows),
        }

    sampled = 0
    seq_sum = 0.0

    for batch in pf.iter_batches(batch_size=min(sample_rows, 1024), columns=[seq_len_col]):
        arr = batch.column(0).to_numpy(zero_copy_only=False)
        if len(arr) == 0:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        seq_sum += float(arr.sum())
        sampled += int(len(arr))
        if sampled >= sample_rows:
            break

    if sampled == 0:
        avg_seq_len = 0.0
    else:
        avg_seq_len = seq_sum / sampled

    cost = float(num_rows) * float(avg_seq_len + 1.0)

    return {
        "num_rows": num_rows,
        "avg_seq_len": float(avg_seq_len),
        "cost": cost,
    }


class ParquetDataset(Dataset):
    """
    小/中型数据集：整体加载到内存
    """
    def __init__(self, data_path, columns=None):
        self.files = _resolve_parquet_files(data_path)
        self.columns = columns
        self.column_index = {}
        self.darray = self.load_data()
        self.num_blocks = len(self.files)
        self.num_samples = int(self.darray.shape[0])

    def __getitem__(self, index):
        return self.darray[index, :]

    def __len__(self):
        return self.darray.shape[0]

    def load_data(self):
        dfs = [pd.read_parquet(fp, columns=self.columns) for fp in self.files]
        df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, axis=0, ignore_index=True)
        darray, column_index = _dataframe_to_darray(df)
        self.column_index = column_index
        return darray


class BlockedParquetBatchDataset(IterableDataset):
    """
    blocked 模式下的 IterableDataset。

    特点：
    - 一个 block 对应一组:
        data/part-xxxxx.parquet
        user_info/part-xxxxx.parquet
        item_info/part-xxxxx.parquet
    - dataset 按 block 读取 data 文件
    - 每次 yield 一个“已经成批”的 payload，避免 batch 内混入不同 block
    - collator 再根据 payload 中的 user/item side-info 文件处理

    DDP 改进：
    - 不再使用简单的 rank::world_size 轮转分配
    - 改为按 block 估计负载(cost)做贪心均衡分配，减少 rank 间严重失衡
    """
    def __init__(self,
                 data_path,
                 user_info_path,
                 item_info_path,
                 columns=None,
                 batch_size=32,
                 shuffle=False,
                 distributed=False,
                 rank=0,
                 world_size=1,
                 drop_last=True):
        super().__init__()
        self.columns = columns
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.epoch = 0

        self.data_part_map = _build_part_file_map(data_path)
        self.user_part_map = _build_part_file_map(user_info_path)
        self.item_part_map = _build_part_file_map(item_info_path)

        common_part_ids = sorted(
            set(self.data_part_map.keys())
            & set(self.user_part_map.keys())
            & set(self.item_part_map.keys())
        )
        if len(common_part_ids) == 0:
            raise ValueError(
                "No matched blocked parquet part ids across data/user_info/item_info.\n"
                f"data parts={sorted(self.data_part_map.keys())}\n"
                f"user parts={sorted(self.user_part_map.keys())}\n"
                f"item parts={sorted(self.item_part_map.keys())}"
            )

        self.blocks = []
        for pid in common_part_ids:
            data_file = self.data_part_map[pid]
            load_stat = _estimate_block_cost(data_file, seq_len_col="seq_len", sample_rows=4096)

            self.blocks.append(
                {
                    "part_id": pid,
                    "data_file": data_file,
                    "user_info_file": self.user_part_map[pid],
                    "item_info_file": self.item_part_map[pid],
                    "num_rows": int(load_stat["num_rows"]),
                    "avg_seq_len": load_stat["avg_seq_len"],
                    "cost": float(load_stat["cost"]),
                }
            )

        if self.distributed:
            self.rank_blocks, self.rank_loads = self._assign_blocks_greedily(self.blocks, self.world_size)
            self.rank_blocks = self.rank_blocks[self.rank]
            self.my_estimated_load = float(self.rank_loads[self.rank])
        else:
            self.rank_blocks = self.blocks
            self.rank_loads = [sum(float(blk["cost"]) for blk in self.blocks)]
            self.my_estimated_load = float(self.rank_loads[0])

        if len(self.rank_blocks) == 0:
            raise ValueError(
                f"No blocked parquet files assigned to rank={rank}. "
                f"total_blocks={len(self.blocks)}, world_size={world_size}"
            )

        self.column_index = self._infer_column_index()
        self.block_row_counts = self._count_rows(self.rank_blocks)
        self.num_blocks = len(self.rank_blocks)
        self.num_samples = int(sum(self.block_row_counts.values()))
        self.num_batches = self._count_batches()

        # 额外统计，便于你打印检查是否均衡
        self.rank_num_rows = int(sum(int(blk["num_rows"]) for blk in self.rank_blocks))
        self.rank_num_cost = float(sum(float(blk["cost"]) for blk in self.rank_blocks))

    def _assign_blocks_greedily(self, blocks, world_size):
        """
        按 block cost 从大到小排序，然后贪心分配给当前总负载最小的 rank。
        """
        sorted_blocks = sorted(blocks, key=lambda x: (x["cost"], x["num_rows"]), reverse=True)

        rank_buckets = [[] for _ in range(world_size)]
        rank_loads = [0.0 for _ in range(world_size)]

        for blk in sorted_blocks:
            target_rank = int(np.argmin(rank_loads))
            rank_buckets[target_rank].append(blk)
            rank_loads[target_rank] += float(blk["cost"])

        # 每个 rank 内部按 part_id 排序，保证稳定性
        for r in range(world_size):
            rank_buckets[r] = sorted(rank_buckets[r], key=lambda x: x["part_id"])

        return rank_buckets, rank_loads

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def _count_rows(self, blocks):
        out = {}
        for blk in blocks:
            out[int(blk["part_id"])] = int(blk["num_rows"])
        return out

    def _count_batches(self):
        total = 0
        for blk in self.rank_blocks:
            n = self.block_row_counts[int(blk["part_id"])]
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += int(np.ceil(n / self.batch_size))
        return int(total)

    def _infer_column_index(self):
        for blk in self.rank_blocks:
            pf = pq.ParquetFile(blk["data_file"])
            for record_batch in pf.iter_batches(batch_size=32, columns=self.columns):
                df = record_batch.to_pandas()
                if len(df) == 0:
                    continue
                _, column_index = _dataframe_to_darray(df)
                return column_index
        raise ValueError("Failed to infer column_index from blocked parquet files.")

    def __iter__(self):
        worker_info = get_worker_info()

        blocks = list(self.rank_blocks)
        py_rng = random.Random(2026 + self.epoch + self.rank)
        np_rng = np.random.default_rng(2026 + self.epoch + self.rank)

        if self.shuffle:
            py_rng.shuffle(blocks)

        if worker_info is not None:
            blocks = blocks[worker_info.id::worker_info.num_workers]

        for blk in blocks:
            df = pd.read_parquet(blk["data_file"], columns=self.columns)
            if len(df) == 0:
                continue

            file_array, _ = _dataframe_to_darray(df)

            if self.shuffle and len(file_array) > 1:
                perm = np_rng.permutation(len(file_array))
                file_array = file_array[perm]

            n = len(file_array)
            start = 0
            while start < n:
                end = min(start + self.batch_size, n)
                if self.drop_last and (end - start) < self.batch_size:
                    break

                rows = np.ascontiguousarray(file_array[start:end])

                yield {
                    "rows": rows,
                    "part_id": int(blk["part_id"]),
                    "data_file": blk["data_file"],
                    "user_info_file": blk["user_info_file"],
                    "item_info_file": blk["item_info_file"],
                }
                start = end


class UniRankDataloader(DataLoader):
    """
    支持两种模式：

    1) 全量模式（blocked=False）:
       - 适用于 KuaiRand_Video_Action / QK_Video_Action
       - 整体读入内存

    2) 分块模式（blocked=True）:
       - 适用于 blocked 数据
       - 自动按 part-xxxxx.parquet 分块读取

    side-info 路径支持两种写法：
    - 新写法（推荐）:
        train_user_info / train_item_info
        valid_user_info / valid_item_info
        test_user_info  / test_item_info
    - 旧写法:
        user_info / item_info
    """
    def __init__(self, feature_map, data_path, user_info=None, item_info=None,
                 batch_size=32, shuffle=False, num_workers=4, max_len=50, padding="pre",
                 distributed=False, rank=0, world_size=1, drop_last=True, split=None, **kwargs):

        self.feature_map = feature_map
        self.split = split

        self.blocked = kwargs.pop("blocked", False)
        self.block_cache_size = kwargs.pop("block_cache_size", 2)

        user_info = _resolve_side_info_path(
            split=self.split,
            key="user_info",
            explicit_path=user_info,
            kwargs=kwargs
        )
        item_info = _resolve_side_info_path(
            split=self.split,
            key="item_info",
            explicit_path=item_info,
            kwargs=kwargs
        )

        sampler = None

        if self.blocked:
            self.dataset = BlockedParquetBatchDataset(
                data_path=data_path,
                user_info_path=user_info,
                item_info_path=item_info,
                columns=None,
                batch_size=batch_size,
                shuffle=shuffle,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                drop_last=drop_last
            )
            if distributed:
                print(
                    f"[Rank {rank}] blocked load balanced: "
                    f"blocks={self.dataset.num_blocks}, "
                    f"rows={self.dataset.rank_num_rows}, "
                    f"batches={self.dataset.num_batches}, "
                    f"est_cost={self.dataset.rank_num_cost:.2f}"
                )
            dataloader_shuffle = False
            actual_batch_size = None
            actual_drop_last = False
            collate_fn = BlockedBatchCollator(
                feature_map=feature_map,
                max_len=max_len,
                column_index=self.dataset.column_index,
                user_info=user_info,
                item_info=item_info,
                padding=padding,
                cache_size=self.block_cache_size
            )
        else:
            self.dataset = ParquetDataset(
                data_path=data_path,
                columns=None
            )

            if distributed:
                sampler = DistributedSampler(
                    self.dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    drop_last=False
                )
            dataloader_shuffle = (shuffle and sampler is None)
            actual_batch_size = batch_size
            actual_drop_last = drop_last
            collate_fn = BatchCollator(
                feature_map=feature_map,
                max_len=max_len,
                column_index=self.dataset.column_index,
                user_info=user_info,
                item_info=item_info,
                padding=padding
            )

        self.sampler_ref = sampler

        super().__init__(
            dataset=self.dataset,
            batch_size=actual_batch_size,
            shuffle=dataloader_shuffle,
            sampler=sampler,
            drop_last=actual_drop_last,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
            collate_fn=collate_fn
        )

        self._configured_batch_size = int(batch_size)
        self.num_blocks = getattr(self.dataset, "num_blocks", 1)

        # 全局样本数，可用于展示
        self.global_num_samples = getattr(self.dataset, "num_samples", len(self.dataset))

        # 当前 rank 实际样本数 / batch 数
        if self.blocked:
            # blocked 模式下 dataset 已经按 rank 切分
            self.num_samples = getattr(self.dataset, "num_samples", len(self.dataset))
            self.num_batches = len(self.dataset)
        else:
            if sampler is not None:
                # DistributedSampler 的长度就是当前 rank 的样本数
                self.num_samples = len(sampler)
            else:
                self.num_samples = len(self.dataset)

            if actual_drop_last:
                self.num_batches = self.num_samples // max(1, self._configured_batch_size)
            else:
                self.num_batches = int(np.ceil(self.num_samples / max(1, self._configured_batch_size)))

    def __len__(self):
        return super().__len__()

    def set_epoch(self, epoch):
        if self.sampler_ref is not None and hasattr(self.sampler_ref, "set_epoch"):
            self.sampler_ref.set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


class BatchCollator(object):
    """
    非 blocked 版本：
    整体 user_info / item_info 读入内存，适合全量数据集。
    """
    def __init__(self, feature_map, max_len, column_index, user_info, item_info, padding="pre"):
        self.feature_map = feature_map
        self.max_len = max_len
        self.padding = padding

        self.all_cols = set(list(feature_map.features.keys()) + feature_map.labels)
        self.batch_cols = [(col, idx) for col, idx in column_index.items() if col in self.all_cols]
        self.task_labels = list(feature_map.labels)

        user_cols = ["user_index", "full_item_seq", "full_action_seq"]
        user_df = pd.read_parquet(user_info, columns=user_cols)

        if "user_index" in user_df.columns:
            user_df = user_df.set_index("user_index").sort_index()
            user_indices = user_df.index.to_numpy(dtype=np.int64)
            self.user_index_min = int(user_indices.min())
            self.user_index_max = int(user_indices.max())

            self.user_row_lookup = torch.full(
                (self.user_index_max - self.user_index_min + 1,),
                -1,
                dtype=torch.long
            )
            self.user_row_lookup[user_indices - self.user_index_min] = torch.arange(
                len(user_df), dtype=torch.long
            )
        else:
            self.user_index_min = None
            self.user_index_max = None
            self.user_row_lookup = None

        self.user_item_seqs = user_df["full_item_seq"].to_numpy()
        self.user_action_seqs = user_df["full_action_seq"].to_numpy()

        meta_fp = _find_meta_data_json(user_info)
        try:
            with open(meta_fp, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        except Exception as e:
            raise ValueError(f"读取 meta_data.json 失败: {meta_fp}, error={e}")

        action_vocab = meta_data.get("action_vocab", None)
        if not action_vocab:
            raise ValueError(
                "meta_data.json 中缺少 action_vocab，"
                "请使用新的 preprocess 文件重新预处理数据。"
            )

        self.action_task_table = self._build_action_task_table(action_vocab)

        item_schema = _get_parquet_schema_names(item_info)
        item_cols = ["item_index"] + [col for col in self.all_cols if col not in {"action", "item_index"}]
        item_cols = [c for c in item_cols if c in item_schema]

        if "item_index" not in item_cols:
            raise ValueError("item_info 中缺少 item_index 列。")

        item_df = pd.read_parquet(item_info, columns=item_cols).set_index("item_index").sort_index()

        if 0 not in item_df.index:
            item_df.loc[0] = 0
            item_df = item_df.sort_index()

        item_indices = item_df.index.to_numpy(dtype=np.int64)
        self.item_index_min = int(item_indices.min())
        self.item_index_max = int(item_indices.max())

        self.item_row_lookup = torch.full(
            (self.item_index_max - self.item_index_min + 1,),
            -1,
            dtype=torch.long
        )
        self.item_row_lookup[item_indices - self.item_index_min] = torch.arange(
            len(item_df), dtype=torch.long
        )

        self.item_tensors = {}
        for col in item_df.columns:
            if col in self.all_cols:
                col_array = np.ascontiguousarray(item_df[col].to_numpy(copy=True))
                self.item_tensors[col] = torch.from_numpy(col_array)

    def _build_action_task_table(self, action_vocab):
        max_action_id = max(int(v) for v in action_vocab.values()) if len(action_vocab) > 0 else 0
        action_task_table = np.zeros(
            (max_action_id + 1, len(self.task_labels)), dtype=np.float32
        )

        def _task_aliases(task_name):
            aliases = {task_name}
            if task_name.startswith("is_"):
                aliases.add(task_name[3:])
            return aliases

        task_alias_sets = [_task_aliases(t) for t in self.task_labels]

        for action_name, action_id in action_vocab.items():
            action_id = int(action_id)
            if action_id <= 0:
                continue
            if not action_name or action_name == "exposure":
                continue
            parts = set(str(action_name).split("|"))
            for t_idx, aliases in enumerate(task_alias_sets):
                if len(parts.intersection(aliases)) > 0:
                    action_task_table[action_id, t_idx] = 1.0

        return torch.from_numpy(action_task_table)

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        batch_dict = _build_batch_dict_from_tensor(batch_tensor, self.batch_cols)

        user_index_tensor = batch_dict["user_index"].long().cpu()

        if self.user_row_lookup is None:
            user_row_ids = user_index_tensor
        else:
            lookup_pos = user_index_tensor - self.user_index_min
            if lookup_pos.min().item() < 0 or lookup_pos.max().item() >= len(self.user_row_lookup):
                raise IndexError(
                    f"user_index 超出 user_info 范围: "
                    f"min={user_index_tensor.min().item()}, max={user_index_tensor.max().item()}, "
                    f"allowed=[{self.user_index_min}, {self.user_index_max}]"
                )
            user_row_ids = self.user_row_lookup[lookup_pos]
            if (user_row_ids < 0).any():
                bad_uid = user_index_tensor[user_row_ids < 0][0].item()
                raise IndexError(f"user_index={bad_uid} 在 user_info 中不存在。")

        user_row_ids = user_row_ids.numpy().astype(np.int64, copy=False)
        seq_lens = batch_dict["seq_len"].int().cpu().numpy()

        user_item_seqs = self.user_item_seqs[user_row_ids]
        user_action_seqs = self.user_action_seqs[user_row_ids]

        batch_item_seqs = self._fast_pad(user_item_seqs, seq_lens)
        batch_action_seqs = self._fast_pad(user_action_seqs, seq_lens)

        mask = torch.from_numpy((batch_item_seqs > 0).astype(np.float32))

        batch_action_tensor = torch.from_numpy(
            batch_action_seqs.astype(np.int64, copy=False)
        )

        token_task_mask = self.action_task_table[batch_action_tensor]
        token_task_mask = token_task_mask * mask.unsqueeze(-1)

        multi_masks = [
            token_task_mask[:, :, t] for t in range(token_task_mask.shape[-1])
        ]

        batch_size = len(user_row_ids)
        seq_total_len = batch_item_seqs.shape[1] + 1

        batch_items = np.empty((batch_size, seq_total_len), dtype=np.int64)
        batch_items[:, :-1] = batch_item_seqs
        batch_items[:, -1] = batch_dict["item_index"].cpu().numpy().astype(np.int64, copy=False)

        batch_actions = np.zeros((batch_size, seq_total_len), dtype=batch_action_seqs.dtype)
        batch_actions[:, :-1] = batch_action_seqs

        flat_items = batch_items.reshape(-1)
        lookup_pos = flat_items - self.item_index_min

        if lookup_pos.min() < 0 or lookup_pos.max() >= len(self.item_row_lookup):
            raise IndexError(
                f"item_index 超出 item_info 范围: "
                f"min={flat_items.min()}, max={flat_items.max()}, "
                f"allowed=[{self.item_index_min}, {self.item_index_max}]"
            )

        lookup_pos_tensor = torch.from_numpy(lookup_pos.astype(np.int64, copy=False))
        row_ids = self.item_row_lookup[lookup_pos_tensor]

        if (row_ids < 0).any():
            bad_item = flat_items[(row_ids < 0).numpy()][0]
            raise IndexError(f"item_index={bad_item} 在 item_info 中不存在。")

        item_dict = {}
        for col, tensor_data in self.item_tensors.items():
            item_dict[col] = tensor_data[row_ids].view(batch_size, seq_total_len)

        if "action" in self.all_cols:
            item_dict["action"] = torch.from_numpy(batch_actions)

        return batch_dict, item_dict, mask, multi_masks

    def _fast_pad(self, user_seqs, seq_lens):
        max_len = self.max_len
        batch_size = len(user_seqs)
        result = np.zeros((batch_size, max_len), dtype=np.int64)

        for i in range(batch_size):
            seq = user_seqs[i]
            l = int(seq_lens[i])
            if l == 0:
                continue

            if self.padding == "pre":
                if l >= max_len:
                    result[i, :] = seq[l - max_len:l]
                else:
                    result[i, max_len - l:] = seq[:l]
            else:
                actual = min(l, max_len)
                result[i, :actual] = seq[:actual]
        return result


class BlockedBatchCollator(object):
    """
    blocked 版本：
    每次只缓存当前 block 的 user_info / item_info，避免一次性读全量 side-info。
    """
    def __init__(self, feature_map, max_len, column_index, user_info, item_info,
                 padding="pre", cache_size=2):
        self.feature_map = feature_map
        self.max_len = max_len
        self.padding = padding
        self.cache_size = int(max(1, cache_size))

        self.all_cols = set(list(feature_map.features.keys()) + feature_map.labels)
        self.batch_cols = [(col, idx) for col, idx in column_index.items() if col in self.all_cols]
        self.task_labels = list(feature_map.labels)

        meta_fp = _find_meta_data_json(user_info)
        try:
            with open(meta_fp, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        except Exception as e:
            raise ValueError(f"读取 meta_data.json 失败: {meta_fp}, error={e}")

        action_vocab = meta_data.get("action_vocab", None)
        if not action_vocab:
            raise ValueError(
                "meta_data.json 中缺少 action_vocab，"
                "无法构造基于 token 的 task-specific multi_masks。"
            )

        self.action_task_table = self._build_action_task_table(action_vocab)
        self.side_cache = OrderedDict()

    def _build_action_task_table(self, action_vocab):
        max_action_id = max(int(v) for v in action_vocab.values()) if len(action_vocab) > 0 else 0
        action_task_table = np.zeros(
            (max_action_id + 1, len(self.task_labels)), dtype=np.float32
        )

        def _task_aliases(task_name):
            aliases = {task_name}
            if task_name.startswith("is_"):
                aliases.add(task_name[3:])
            return aliases

        task_alias_sets = [_task_aliases(t) for t in self.task_labels]

        for action_name, action_id in action_vocab.items():
            action_id = int(action_id)
            if action_id <= 0:
                continue
            if not action_name or action_name == "exposure":
                continue
            parts = set(str(action_name).split("|"))
            for t_idx, aliases in enumerate(task_alias_sets):
                if len(parts.intersection(aliases)) > 0:
                    action_task_table[action_id, t_idx] = 1.0

        return torch.from_numpy(action_task_table)

    def _load_block_side_info(self, user_info_file, item_info_file):
        cache_key = (str(user_info_file), str(item_info_file))
        if cache_key in self.side_cache:
            value = self.side_cache.pop(cache_key)
            self.side_cache[cache_key] = value
            return value

        need_user_cols = ["user_index", "full_item_seq", "full_action_seq"]
        user_df = pd.read_parquet(user_info_file, columns=need_user_cols)
        user_df = user_df.set_index("user_index").sort_index()

        if len(user_df) == 0:
            raise ValueError(f"Empty blocked user_info file: {user_info_file}")

        user_indices = user_df.index.to_numpy(dtype=np.int64)
        user_index_min = int(user_indices.min())
        user_index_max = int(user_indices.max())

        user_row_lookup = torch.full(
            (user_index_max - user_index_min + 1,),
            -1,
            dtype=torch.long
        )
        user_row_lookup[user_indices - user_index_min] = torch.arange(
            len(user_df), dtype=torch.long
        )

        user_item_seqs = user_df["full_item_seq"].to_numpy()
        user_action_seqs = user_df["full_action_seq"].to_numpy()

        item_schema = _get_parquet_schema_names(item_info_file)
        item_cols = ["item_index"] + [col for col in self.all_cols if col not in {"action", "item_index"}]
        item_cols = [c for c in item_cols if c in item_schema]

        if "item_index" not in item_cols:
            raise ValueError(f"blocked item_info 中缺少 item_index 列: {item_info_file}")

        item_df = pd.read_parquet(item_info_file, columns=item_cols).set_index("item_index").sort_index()

        if 0 not in item_df.index:
            item_df.loc[0] = 0
            item_df = item_df.sort_index()

        item_indices = item_df.index.to_numpy(dtype=np.int64)
        item_index_min = int(item_indices.min())
        item_index_max = int(item_indices.max())

        item_row_lookup = torch.full(
            (item_index_max - item_index_min + 1,),
            -1,
            dtype=torch.long
        )
        item_row_lookup[item_indices - item_index_min] = torch.arange(
            len(item_df), dtype=torch.long
        )

        item_tensors = {}
        for col in item_df.columns:
            if col in self.all_cols:
                col_array = np.ascontiguousarray(item_df[col].to_numpy(copy=True))
                item_tensors[col] = torch.from_numpy(col_array)

        side_info = {
            "user_index_min": user_index_min,
            "user_index_max": user_index_max,
            "user_row_lookup": user_row_lookup,
            "user_item_seqs": user_item_seqs,
            "user_action_seqs": user_action_seqs,
            "item_index_min": item_index_min,
            "item_index_max": item_index_max,
            "item_row_lookup": item_row_lookup,
            "item_tensors": item_tensors,
        }

        self.side_cache[cache_key] = side_info
        while len(self.side_cache) > self.cache_size:
            self.side_cache.popitem(last=False)

        return side_info

    def __call__(self, payload):
        if not isinstance(payload, dict):
            raise TypeError(
                "BlockedBatchCollator expects dataset payload dict, "
                f"but got type={type(payload)}"
            )

        rows = payload["rows"]
        user_info_file = payload["user_info_file"]
        item_info_file = payload["item_info_file"]

        if not isinstance(rows, np.ndarray):
            rows = np.asarray(rows)

        batch_tensor = torch.from_numpy(rows)
        batch_dict = _build_batch_dict_from_tensor(batch_tensor, self.batch_cols)

        side = self._load_block_side_info(user_info_file, item_info_file)

        user_index_tensor = batch_dict["user_index"].long().cpu()
        lookup_pos = user_index_tensor - side["user_index_min"]

        if lookup_pos.min().item() < 0 or lookup_pos.max().item() >= len(side["user_row_lookup"]):
            raise IndexError(
                f"user_index 超出 blocked user_info 范围: "
                f"min={user_index_tensor.min().item()}, max={user_index_tensor.max().item()}, "
                f"allowed=[{side['user_index_min']}, {side['user_index_max']}] | file={user_info_file}"
            )

        user_row_ids = side["user_row_lookup"][lookup_pos]
        if (user_row_ids < 0).any():
            bad_uid = user_index_tensor[user_row_ids < 0][0].item()
            raise IndexError(f"user_index={bad_uid} 在 blocked user_info 中不存在。file={user_info_file}")

        user_row_ids = user_row_ids.numpy().astype(np.int64, copy=False)
        seq_lens = batch_dict["seq_len"].int().cpu().numpy()

        user_item_seqs = side["user_item_seqs"][user_row_ids]
        user_action_seqs = side["user_action_seqs"][user_row_ids]

        batch_item_seqs = self._fast_pad(user_item_seqs, seq_lens)
        batch_action_seqs = self._fast_pad(user_action_seqs, seq_lens)

        mask = torch.from_numpy((batch_item_seqs > 0).astype(np.float32))

        batch_action_tensor = torch.from_numpy(
            batch_action_seqs.astype(np.int64, copy=False)
        )

        token_task_mask = self.action_task_table[batch_action_tensor]
        token_task_mask = token_task_mask * mask.unsqueeze(-1)
        multi_masks = [token_task_mask[:, :, t] for t in range(token_task_mask.shape[-1])]

        batch_size = len(user_row_ids)
        seq_total_len = batch_item_seqs.shape[1] + 1

        batch_items = np.empty((batch_size, seq_total_len), dtype=np.int64)
        batch_items[:, :-1] = batch_item_seqs
        batch_items[:, -1] = batch_dict["item_index"].cpu().numpy().astype(np.int64, copy=False)

        batch_actions = np.zeros((batch_size, seq_total_len), dtype=batch_action_seqs.dtype)
        batch_actions[:, :-1] = batch_action_seqs

        flat_items = batch_items.reshape(-1)
        lookup_pos = flat_items - side["item_index_min"]

        if lookup_pos.min() < 0 or lookup_pos.max() >= len(side["item_row_lookup"]):
            raise IndexError(
                f"item_index 超出 blocked item_info 范围: "
                f"min={flat_items.min()}, max={flat_items.max()}, "
                f"allowed=[{side['item_index_min']}, {side['item_index_max']}] | file={item_info_file}"
            )

        lookup_pos_tensor = torch.from_numpy(lookup_pos.astype(np.int64, copy=False))
        row_ids = side["item_row_lookup"][lookup_pos_tensor]

        if (row_ids < 0).any():
            bad_item = flat_items[(row_ids < 0).numpy()][0]
            raise IndexError(f"item_index={bad_item} 在 blocked item_info 中不存在。file={item_info_file}")

        item_dict = {}
        for col, tensor_data in side["item_tensors"].items():
            item_dict[col] = tensor_data[row_ids].view(batch_size, seq_total_len)

        if "action" in self.all_cols:
            item_dict["action"] = torch.from_numpy(batch_actions)

        return batch_dict, item_dict, mask, multi_masks

    def _fast_pad(self, user_seqs, seq_lens):
        max_len = self.max_len
        batch_size = len(user_seqs)
        result = np.zeros((batch_size, max_len), dtype=np.int64)

        for i in range(batch_size):
            seq = user_seqs[i]
            l = int(seq_lens[i])
            if l == 0:
                continue

            if self.padding == "pre":
                if l >= max_len:
                    result[i, :] = seq[l - max_len:l]
                else:
                    result[i, max_len - l:] = seq[:l]
            else:
                actual = min(l, max_len)
                result[i, :actual] = seq[:actual]
        return result