#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_QK_seq_action.py — QK-Video (内存优化版, 分块分区处理)
===============================================================
参考 KuaiRand-27K 分块处理方案，通过「用户哈希分区」避免内存溢出 (OOM)。

切分策略（按用户行为序列比例 8:1:1）:
  - 每个用户内部按原始行为顺序切分
  - 前 ~80% → 训练集
  - 中间 ~10% → 验证集
  - 后 ~10% → 测试集

处理流程:
  Phase 1 — CSV 分块读取 → 清洗 → 按用户哈希分区写入 Parquet 临时文件
             同时收集全局统计信息（用户交互数、物品集、特征唯一值）
  Phase 2 — 逐分区: 编码特征 + 按用户序列比例切分 + ParquetWriter 增量写出
  Phase 3 — 汇总保存 user_info / item_info / meta_data + 清理临时文件

本版本说明:
  - 不再构建 user_info.behavior_type_mask
  - 改为在 meta_data.json 中保存 action_vocab
  - 供 dataloader 在训练时基于 full_action_seq 构造 task-specific token masks

内存峰值 ≈ 单分区大小 + 写缓冲，远小于全量加载。

依赖:
    pip install pandas numpy pyarrow

用法:
    python preprocess_QK_seq_action.py
"""

import gc
import json
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ================================================================
#  1. 常量 & 列定义
# ================================================================

REQUIRED_COLUMNS = [
    "user_id", "item_id", "click", "follow", "like", "share",
    "video_category", "watching_times", "gender", "age",
]

LABEL_COLUMNS = ["click", "follow", "like", "share"]

FINAL_COLUMNS = [
    "user_index", "item_index", "seq_len", "user_id",
    "gender", "age", "watching_times",
    "click", "follow", "like", "share",
]

CAT_COLUMNS = ["user_id", "item_id", "video_category", "watching_times", "gender", "age"]

# 预计算 action 查找表: 4 个二值 label 共 2^4 = 16 种组合
_ACTION_LOOKUP = np.empty(16, dtype=object)
for _k in range(16):
    if _k == 0:
        _ACTION_LOOKUP[_k] = "exposure"
    else:
        _ACTION_LOOKUP[_k] = "|".join(
            col for bit, col in zip([8, 4, 2, 1], LABEL_COLUMNS) if _k & bit
        )


# ================================================================
#  2. 工具函数
# ================================================================

def check_required_columns(columns):
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"数据集中缺少以下列: {missing}")


def build_ordered_vocab(values, start=1):
    """从可迭代对象构建 vocab，从 start 开始编码，0 预留给 padding/unknown。"""
    uniq = list(dict.fromkeys(str(v) for v in values))
    return {v: i for i, v in enumerate(uniq, start=start)}


def _allocate_split_counts(n, train_ratio, valid_ratio, test_ratio):
    """将长度为 n 的单个用户行为序列按比例分配到 train/valid/test，每个至少 1 条。"""
    if n < 3:
        raise ValueError(f"单个用户行为数必须 >= 3，当前 n={n}")
    ratios = np.array([train_ratio, valid_ratio, test_ratio], dtype=np.float64)
    raw = ratios * n
    counts = np.floor(raw).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for i in range(remainder):
            counts[order[i % 3]] += 1
    for idx in range(3):
        while counts[idx] == 0:
            donors = np.where(counts > 1)[0]
            if len(donors) == 0:
                raise ValueError(
                    f"无法为每个 split 分配至少 1 条样本: n={n}, counts={counts.tolist()}"
                )
            donor = donors[np.argmax(counts[donors])]
            counts[donor] -= 1
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def check_static_consistency(df, key_col, value_cols, name):
    """检查静态特征是否一致，若不一致则 warning。"""
    inconsistent = {}
    for col in value_cols:
        nunique = df.groupby(key_col, sort=False)[col].nunique(dropna=False)
        bad = int((nunique > 1).sum())
        if bad > 0:
            inconsistent[col] = bad
    if inconsistent:
        warnings.warn(
            f"{name} 存在同一 key 对应多个取值的情况，"
            f"将保留最后一次出现的值: {inconsistent}"
        )


# ================================================================
#  3. Phase 1: CSV → Partitioned Parquet + 全局统计
# ================================================================

def _clean_chunk(chunk: pd.DataFrame, global_row_offset: int):
    """
    对单个 chunk 执行基础清洗:
      1. 仅保留必要列
      2. label → float32
      3. categorical → clean str
      4. 丢弃 user_id / item_id 缺失行
      5. 构造 exposure + action (向量化查表)
      6. 分配全局 _row_id
    """
    chunk = chunk[REQUIRED_COLUMNS].copy()

    # label -> float32
    for col in LABEL_COLUMNS:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0).astype(np.float32)

    # categorical -> clean str
    for col in CAT_COLUMNS:
        original_na = chunk[col].isna()
        s = chunk[col].astype(str)
        bad = original_na | (s.str.strip() == "")
        chunk[col] = np.where(bad, "__MISSING__", s.values)

    # 丢弃 user_id / item_id 缺失行
    mask = (chunk["user_id"] != "__MISSING__") & (chunk["item_id"] != "__MISSING__")
    chunk = chunk[mask].reset_index(drop=True)

    if len(chunk) == 0:
        return chunk, global_row_offset

    # 构造 exposure + action (向量化查表，替代逐行 apply)
    label_vals = np.column_stack([chunk[c].values for c in LABEL_COLUMNS])
    binary = (label_vals > 0).astype(np.uint8)
    chunk["exposure"] = (binary.sum(axis=1) == 0).astype(np.float32)
    keys = binary[:, 0] * 8 + binary[:, 1] * 4 + binary[:, 2] * 2 + binary[:, 3]
    chunk["action"] = _ACTION_LOOKUP[keys]
    del label_vals, binary, keys

    # 全局 _row_id (保持跨 chunk 的原始行顺序)
    n = len(chunk)
    chunk["_row_id"] = np.arange(
        global_row_offset, global_row_offset + n, dtype=np.int64
    )
    return chunk, global_row_offset + n


def phase1_partition_to_parquet(
    input_file: str,
    tmp_dir: Path,
    n_parts: int,
    chunk_size: int,
    buffer_flush_size: int,
):
    """
    逐 chunk 读取 CSV → 清洗 → 按 user_id 哈希分区写入 Parquet。
    同时收集:
      - user_counts: {user_id_str: 交互数}
      - item_ids:    set of all item_id_str
      - unique_values: {feature_name: [按首次出现排序的唯一值]}
    """
    print(
        f"\n[Phase 1] CSV → 分区 Parquet "
        f"(n_parts={n_parts}, chunk_size={chunk_size:,})"
    )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in range(n_parts):
        (tmp_dir / f"part_{p:03d}").mkdir(exist_ok=True)

    # 检查列是否齐全
    sample = pd.read_csv(input_file, nrows=5)
    check_required_columns(sample.columns)
    del sample

    # ---- 全局统计容器 ----
    user_counts = defaultdict(int)
    item_ids_set = set()

    # 按首次出现顺序追踪唯一值
    unique_trackers = {
        "video_category": {},
        "watching_times": {},
        "gender": {},
        "age": {},
        "action": {},
    }

    # ---- 分区写缓冲 ----
    buffers = {p: [] for p in range(n_parts)}
    buf_sizes = {p: 0 for p in range(n_parts)}
    file_counts = {p: 0 for p in range(n_parts)}

    def flush(pid):
        if not buffers[pid]:
            return
        out = pd.concat(buffers[pid], ignore_index=True)
        fp = tmp_dir / f"part_{pid:03d}" / f"c_{file_counts[pid]:04d}.parquet"
        out.to_parquet(fp, index=False, engine="pyarrow")
        file_counts[pid] += 1
        buffers[pid] = []
        buf_sizes[pid] = 0
        del out

    global_row_offset = 0
    total_rows = 0

    reader = pd.read_csv(input_file, chunksize=chunk_size)

    for chunk_idx, raw_chunk in enumerate(reader):
        chunk, global_row_offset = _clean_chunk(raw_chunk, global_row_offset)
        del raw_chunk

        if len(chunk) == 0:
            continue

        # ---- 收集统计 ----
        vc = chunk["user_id"].value_counts(sort=False)
        for uid, cnt in zip(vc.index, vc.values):
            user_counts[uid] += int(cnt)
        item_ids_set.update(chunk["item_id"].unique().tolist())

        for feat_name in unique_trackers:
            for v in chunk[feat_name].unique():
                unique_trackers[feat_name].setdefault(v, None)

        # ---- 按 user_id 哈希分区路由 ----
        hash_vals = pd.util.hash_pandas_object(
            chunk["user_id"], index=False
        ).values
        part_arr = (hash_vals % n_parts).astype(np.int32)

        for pid in range(n_parts):
            mask = part_arr == pid
            n_match = int(mask.sum())
            if n_match > 0:
                buffers[pid].append(chunk.loc[mask].copy())
                buf_sizes[pid] += n_match
                if buf_sizes[pid] >= buffer_flush_size:
                    flush(pid)

        total_rows += len(chunk)
        del chunk, part_arr, hash_vals
        gc.collect()

        if (chunk_idx + 1) % 10 == 0:
            print(f"    chunk {chunk_idx + 1}: 累计 {total_rows:,} 行")

    # ---- 最终刷盘 ----
    for pid in range(n_parts):
        flush(pid)

    del buffers, buf_sizes
    gc.collect()

    unique_values = {k: list(v.keys()) for k, v in unique_trackers.items()}

    print(
        f"  完成: {total_rows:,} 行, "
        f"{len(user_counts):,} 用户, {len(item_ids_set):,} 物品\n"
    )
    return dict(user_counts), item_ids_set, unique_values


# ================================================================
#  4. Phase 2: Per-partition Process + Incremental Write
# ================================================================

def process_partition(
    tmp_dir: Path,
    pid: int,
    valid_users: set,
    user_idx_map: dict,
    item_idx_map: dict,
    vocabs: dict,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
):
    """
    处理一个用户分区:
      1. 读取分区下所有 parquet 文件
      2. 过滤有效用户
      3. 按 (user_id, _row_id) 排序 → 保持原始行为顺序
      4. 编码 user_index / item_index / categorical features / action
      5. 按用户序列比例切分 train/valid/test
      6. 构建 user_info / item_info 片段

    返回 dict{train, valid, test, user_info, item_info} 或 None。
    """
    part_dir = tmp_dir / f"part_{pid:03d}"
    files = sorted(part_dir.glob("*.parquet"))
    if not files:
        return None

    # ---- 读取 + 过滤 ----
    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        d = d[d["user_id"].isin(valid_users)]
        if len(d) > 0:
            dfs.append(d)
        del d
    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    # ---- 排序: 用户内按 _row_id 保持原始行为顺序 ----
    df.sort_values(["user_id", "_row_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---- 用户/物品静态特征一致性检查 ----
    check_static_consistency(
        df, key_col="user_id", value_cols=["gender", "age"],
        name=f"user side (partition {pid})",
    )
    check_static_consistency(
        df, key_col="item_id", value_cols=["video_category"],
        name=f"item side (partition {pid})",
    )

    # ---- 编码 ID ----
    df["user_index"] = df["user_id"].map(user_idx_map).astype(np.int32)
    df["item_index"] = df["item_id"].map(item_idx_map).fillna(0).astype(np.int32)

    # user_id categorical id: 1-based
    df["user_id"] = (df["user_index"] + 1).astype(np.int32)

    # ---- 编码 categorical features ----
    for col in ["video_category", "watching_times", "gender", "age"]:
        df[col] = df[col].map(vocabs[col]).fillna(0).astype(np.int32)
    df["action"] = df["action"].map(vocabs["action"]).fillna(0).astype(np.int32)

    # ---- seq_len: 当前行为之前已有多少条历史行为 ----
    df["seq_len"] = df.groupby("user_index", sort=False).cumcount().astype(np.int32)

    # ---- 收集 item_info 片段: {item_index → encoded video_category} ----
    item_info_frag = {}
    item_sub = df[["item_index", "video_category"]].drop_duplicates(
        "item_index", keep="last"
    )
    for idx, vc in zip(item_sub["item_index"].values, item_sub["video_category"].values):
        item_info_frag[int(idx)] = int(vc)
    del item_sub

    # ---- 构建 user_info 片段 ----
    user_info_rows = []
    for uidx, gdf in df.groupby("user_index", sort=True):
        user_info_rows.append(
            {
                "user_index": int(uidx),
                "full_item_seq": gdf["item_index"].tolist(),
                "full_action_seq": gdf["action"].tolist(),
            }
        )

    # ---- 按用户序列比例切分 ----
    g = df.groupby("user_index", sort=False)
    cum_pos = g.cumcount().values
    user_sizes = g["_row_id"].transform("size").values

    unique_sizes = np.unique(user_sizes)
    max_size = int(unique_sizes.max())
    train_end_lut = np.zeros(max_size + 1, dtype=np.int32)
    valid_end_lut = np.zeros(max_size + 1, dtype=np.int32)

    for n in unique_sizes:
        n = int(n)
        tc, vc, _ = _allocate_split_counts(n, train_ratio, valid_ratio, test_ratio)
        train_end_lut[n] = tc
        valid_end_lut[n] = tc + vc

    train_end = train_end_lut[user_sizes]
    valid_end = valid_end_lut[user_sizes]

    split_arr = np.full(len(df), 2, dtype=np.int8)  # 2=test
    split_arr[cum_pos < valid_end] = 1  # 1=valid
    split_arr[cum_pos < train_end] = 0  # 0=train

    def _select_final(split_code):
        mask = split_arr == split_code
        if mask.sum() == 0:
            return pd.DataFrame(columns=FINAL_COLUMNS)
        return df.loc[mask, FINAL_COLUMNS].reset_index(drop=True)

    result = {
        "train": _select_final(0),
        "valid": _select_final(1),
        "test": _select_final(2),
        "user_info": user_info_rows,
        "item_info": item_info_frag,
    }

    del df, cum_pos, user_sizes, split_arr, train_end, valid_end
    gc.collect()
    return result


# ================================================================
#  5. 主流程
# ================================================================

def preprocess_and_split(
    input_file="QK-video.csv",
    output_dir="./data/QK_Video",
    min_user_interactions=3,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    n_user_parts=20,
    chunk_size=1_000_000,
    buffer_flush_size=300_000,
):
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"train_ratio + valid_ratio + test_ratio 必须等于 1，当前为 {ratio_sum}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp_partitions"

    # ================================================================
    #  Step 1/6: Phase 1 — CSV → 分区 Parquet + 全局统计
    # ================================================================
    user_counts, item_ids_set, unique_values = phase1_partition_to_parquet(
        input_file=input_file,
        tmp_dir=tmp_dir,
        n_parts=n_user_parts,
        chunk_size=chunk_size,
        buffer_flush_size=buffer_flush_size,
    )
    gc.collect()

    # ================================================================
    #  Step 2/6: 过滤低频用户 + 构建全局 ID 映射 + Vocab
    # ================================================================
    print("[Step 2/6] 过滤低频用户 + 构建全局映射 + Vocab")

    valid_users = {
        u for u, c in user_counts.items() if c >= min_user_interactions
    }
    n_dropped = len(user_counts) - len(valid_users)
    dropped_rows = sum(
        c for u, c in user_counts.items() if c < min_user_interactions
    )
    if n_dropped > 0:
        print(
            f"  [Info] 过滤交互数 < {min_user_interactions} 的用户: "
            f"users={n_dropped:,}, rows={dropped_rows:,}"
        )
    print(f"  有效用户: {len(valid_users):,}")

    if not valid_users:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError("过滤后数据为空，请检查原始数据或调小 min_user_interactions。")

    # user_idx_map: 0-based (sorted for determinism)
    sorted_users = sorted(valid_users)
    user_idx_map = {u: i for i, u in enumerate(sorted_users)}

    # item_idx_map: 1-based, 0 = padding
    sorted_items = sorted(item_ids_set)
    item_idx_map = {it: i + 1 for i, it in enumerate(sorted_items)}
    print(f"  物品数:   {len(item_idx_map):,}")

    # 构建 vocab (1-based, 0=padding/unknown)
    vocabs = {}
    for feat_name, uv in unique_values.items():
        vocabs[feat_name] = build_ordered_vocab(uv, start=1)

    # vocab_size (含 padding 位)
    vocab_size = {
        "user_index": len(user_idx_map),
        "item_index": len(item_idx_map) + 1,
        "user_id": len(user_idx_map) + 1,
        "item_id": len(item_idx_map) + 1,
        "video_category": len(vocabs["video_category"]) + 1,
        "watching_times": len(vocabs["watching_times"]) + 1,
        "gender": len(vocabs["gender"]) + 1,
        "age": len(vocabs["age"]) + 1,
        "action": len(vocabs["action"]) + 1,
    }
    print(f"  action 种类: {len(vocabs['action'])}")

    del user_counts, item_ids_set, unique_values, sorted_users, sorted_items
    gc.collect()

    # ================================================================
    #  Step 3/6: Phase 2 — 逐分区编码 + 切分 + 增量写出
    # ================================================================
    print(f"\n[Step 3/6] 逐分区编码 + 按用户序列比例切分 + 增量写出")

    writers = {s: None for s in ["train", "valid", "test"]}
    all_user_info = []
    all_item_info = {}  # {item_index: video_category}
    sample_counts = {"train": 0, "valid": 0, "test": 0}
    max_seq = 0

    for pid in range(n_user_parts):
        result = process_partition(
            tmp_dir=tmp_dir,
            pid=pid,
            valid_users=valid_users,
            user_idx_map=user_idx_map,
            item_idx_map=item_idx_map,
            vocabs=vocabs,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
        )
        if result is None:
            continue

        # 增量写出 train/valid/test
        for split_name in ["train", "valid", "test"]:
            sdf = result[split_name]
            if len(sdf) == 0:
                continue
            table = pa.Table.from_pandas(sdf, preserve_index=False)
            if writers[split_name] is None:
                writers[split_name] = pq.ParquetWriter(
                    str(output_dir / f"{split_name}.parquet"), table.schema
                )
            writers[split_name].write_table(table)
            sample_counts[split_name] += len(sdf)

        # 收集 user_info 片段
        for row in result["user_info"]:
            sl = len(row["full_item_seq"])
            if sl > max_seq:
                max_seq = sl
        all_user_info.extend(result["user_info"])

        # 收集 item_info 片段
        all_item_info.update(result["item_info"])

        done = sum(sample_counts.values())
        print(f"  partition {pid + 1:3d}/{n_user_parts}: 累计 {done:,} 行")

        del result
        gc.collect()

    # 关闭 writers
    for w in writers.values():
        if w is not None:
            w.close()

    total = sum(sample_counts.values())
    print(
        f"\n  train={sample_counts['train']:,}  "
        f"valid={sample_counts['valid']:,}  "
        f"test={sample_counts['test']:,}"
    )
    print(f"  总计={total:,}")

    if total == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError("处理后数据为空，请检查原始数据或调整参数。")
    for s in ["train", "valid", "test"]:
        if sample_counts[s] == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise ValueError(f"{s} 集为空，请检查数据或调整切分参数。")

    print(
        f"  切分比例(目标): train={train_ratio:.2f}, "
        f"valid={valid_ratio:.2f}, test={test_ratio:.2f}"
    )

    # 这里稍后还要用 vocabs["action"] 写入 meta_data，先不删 vocabs
    del valid_users, user_idx_map, item_idx_map
    gc.collect()

    # ================================================================
    #  Step 4/6: 保存 user_info
    # ================================================================
    print(f"\n[Step 4/6] 保存 user_info")
    num_users = vocab_size["user_index"]
    ui_dict = {r["user_index"]: r for r in all_user_info}

    offsets = [0]
    item_seqs_flat = []
    action_seqs_flat = []

    for i in range(num_users):
        row = ui_dict.get(i)
        if row is not None:
            iseq = row["full_item_seq"]
            aseq = row["full_action_seq"]
            item_seqs_flat.extend(iseq)
            action_seqs_flat.extend(aseq)
            offsets.append(offsets[-1] + len(iseq))
        else:
            offsets.append(offsets[-1])

    pa_offsets = pa.array(offsets, type=pa.int64())
    table = pa.table(
        {
            "user_index": pa.array(np.arange(num_users, dtype=np.int32)),
            "full_item_seq": pa.ListArray.from_arrays(
                pa_offsets, pa.array(item_seqs_flat, type=pa.int32())
            ),
            "full_action_seq": pa.ListArray.from_arrays(
                pa_offsets, pa.array(action_seqs_flat, type=pa.int32())
            ),
        }
    )
    pq.write_table(table, output_dir / "user_info.parquet")
    print(f"  [Saved] user_info.parquet ({num_users:,} users)")

    del (
        all_user_info, ui_dict, item_seqs_flat, action_seqs_flat,
        offsets, table
    )
    gc.collect()

    # ================================================================
    #  Step 5/6: 保存 item_info
    # ================================================================
    print(f"\n[Step 5/6] 保存 item_info")
    num_items = vocab_size["item_index"]  # 包含 padding (index 0)

    item_id_arr = np.zeros(num_items, dtype=np.int32)
    video_cat_arr = np.zeros(num_items, dtype=np.int32)

    for idx, vc in all_item_info.items():
        if 0 < idx < num_items:
            item_id_arr[idx] = idx  # item_id = item_index
            video_cat_arr[idx] = vc

    table = pa.table(
        {
            "item_index": pa.array(np.arange(num_items, dtype=np.int32)),
            "item_id": pa.array(item_id_arr),
            "video_category": pa.array(video_cat_arr),
        }
    )
    pq.write_table(table, output_dir / "item_info.parquet")
    print(f"  [Saved] item_info.parquet ({num_items:,} items incl. padding)")

    del all_item_info, item_id_arr, video_cat_arr, table
    gc.collect()

    # ================================================================
    #  Step 6/6: 保存 meta_data.json + 清理
    # ================================================================
    print(f"\n[Step 6/6] 保存 meta_data + 清理临时文件")

    meta_data = {
        "sample_size": {
            "total": total,
            "train": sample_counts["train"],
            "valid": sample_counts["valid"],
            "test": sample_counts["test"],
        },
        "vocab_size": {k: int(v) for k, v in vocab_size.items()},
        "label": LABEL_COLUMNS,
        "action_vocab": {k: int(v) for k, v in vocabs["action"].items()},
        "action_vocab_desc": (
            "编码后的 action 词表，用于 dataloader 基于 full_action_seq "
            "构造 task-specific token masks。"
        ),
        "max_len": {
            "full_item_seq": max_seq,
            "full_action_seq": max_seq,
        },
    }

    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)
    print(f"  [Saved] meta_data.json")

    del vocabs
    gc.collect()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  临时分区文件已清理")

    # ================================================================
    #  Summary
    # ================================================================
    print("\n" + "=" * 55)
    print("  QK-Video Preprocess Done (分块分区处理)")
    print("=" * 55)
    print(f"输出目录: {output_dir}\n")
    for fname in [
        "train.parquet", "valid.parquet", "test.parquet",
        "user_info.parquet", "item_info.parquet", "meta_data.json",
    ]:
        print(f"  ✔ {output_dir / fname}")
    print("\nmeta_data.json:")
    print(json.dumps(meta_data, ensure_ascii=False, indent=4))


# ================================================================

if __name__ == "__main__":
    preprocess_and_split(
        input_file="./QK-video.csv",
        output_dir="../QK_Video_Action",
        min_user_interactions=10,
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
        n_user_parts=10,
        chunk_size=2_000_000,
        buffer_flush_size=500_000,
    )