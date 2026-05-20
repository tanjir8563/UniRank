#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_TencentGR_blocked_seq_action.py
==========================================

将 TencentGR / TAAC2025 Seq-Action 预处理与 block 切分合一，直接生成：

output_dir/
  train/
    data/part-00000.parquet
    user_info/part-00000.parquet
    item_info/part-00000.parquet
    ...
  valid/
    data/part-00000.parquet
    user_info/part-00000.parquet
    item_info/part-00000.parquet
    ...
  test/
    data/part-00000.parquet
    user_info/part-00000.parquet
    item_info/part-00000.parquet
    ...
  meta_data.json
  block_manifest.json

核心设计
--------
1. 先按 user_id hash 把 seq.parquet 分区，避免一次性爆内存
2. 再逐个 partition 处理，按时间顺序 + 正样本约束切 train/valid/test
3. 每个 split 再分配到多个 block
4. 每个 block 单独生成自己的：
   - data
   - user_info
   - item_info
5. block 内部的 user_index / item_index 重新映射为局部连续 id
   - user_id / item_id 保持全局 id，不影响 embedding 一致性

新增规则
--------
1. 若一个用户完整行为序列全是负样本（既无 click 也无 conversion），则过滤该用户
2. test 为最后一个“至少包含 click/conversion 正样本”的最短时间段
3. valid 为紧挨着 test 的、至少包含 click/conversion 正样本的最短时间段
4. 剩余全部给 train
5. 统计 vocab_size 时，显式加入 item static features
"""

import argparse
import gc
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ================================================================
# 1. 常量定义
# ================================================================

BEIJING_TZ = "Asia/Shanghai"

USER_SCALAR_FEATURES = ["103", "104", "105", "109"]
USER_LIST_FEATURES = ["106", "107", "108", "110"]
USER_LIST_KEEP = 5

ITEM_STATIC_FEATURES = [
    "100", "101", "102", "112", "114", "115", "116",
    "117", "118", "119", "120", "121", "122"
]

LABEL_COLUMNS = ["is_click", "is_conversion"]
CONTEXT_FEATURES = ["day_of_week", "is_weekend", "hour"]

FINAL_COLUMNS = (
    ["user_index", "item_index", "seq_len", "user_id", "timestamp"]
    + USER_SCALAR_FEATURES
    + USER_LIST_FEATURES
    + CONTEXT_FEATURES
    + LABEL_COLUMNS
)


# ================================================================
# 2. 工具函数
# ================================================================

def prepare_output_dir(output_dir: Path, overwrite: bool = False):
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                f"Please remove it first, or use --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def ensure_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c) for c in df.columns]
    return df


def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def normalize_list_feature(v, keep=5):
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = [safe_int(x, 0) for x in list(v)[:keep]]
    else:
        arr = []
    if len(arr) < keep:
        arr += [0] * (keep - len(arr))
    return arr


def detect_timestamp_unit_from_sample(values) -> str:
    for v in values:
        try:
            x = float(v)
            if not math.isnan(x):
                return "ms" if x >= 1e12 else "s"
        except Exception:
            continue
    return "ms"


def fmt_date_int(d: int) -> str:
    return f"{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}"


def build_action_maps():
    raw2name = {
        0: "exposure",
        1: "click",
        2: "conversion",
    }
    name2code = {name: i + 1 for i, name in enumerate(sorted(set(raw2name.values())))}
    return raw2name, name2code


def get_scalar_vocab_size(series: pd.Series) -> int:
    if len(series) == 0:
        return 1
    mx = pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int64).max()
    return int(mx) + 1


def get_list_vocab_size(series: pd.Series) -> int:
    mx = 0
    for arr in series:
        if isinstance(arr, (list, tuple, np.ndarray)):
            for x in arr:
                xi = safe_int(x, 0)
                if xi > mx:
                    mx = xi
    return int(mx) + 1


def count_positive_users_from_action_counter(user_action_counter):
    kept = 0
    dropped = 0
    kept_users = set()
    for uid, mp in user_action_counter.items():
        click_pos = int(mp.get(1, 0))
        conv_pos = int(mp.get(2, 0))
        if click_pos > 0 or conv_pos > 0:
            kept += 1
            kept_users.add(int(uid))
        else:
            dropped += 1
    return kept_users, kept, dropped


def has_required_positive(date_list, date_action_counter,
                          require_click_positive=True,
                          require_conversion_positive=True):
    click_pos = 0
    conv_pos = 0
    for d in date_list:
        mp = date_action_counter.get(int(d), {})
        click_pos += int(mp.get(1, 0))
        conv_pos += int(mp.get(2, 0))

    ok = True
    if require_click_positive:
        ok = ok and (click_pos > 0)
    if require_conversion_positive:
        ok = ok and (conv_pos > 0)

    return ok, click_pos, conv_pos


def build_minimal_tail_splits(sorted_dates,
                              date_action_counter,
                              require_click_positive=True,
                              require_conversion_positive=True):
    """
    固定规则：
    1) test 为最后一个“至少包含正样本”的最短时间段
    2) valid 为紧挨着 test 的、至少包含正样本的最短时间段
    3) 剩余全部给 train
    """
    n_days = len(sorted_dates)
    if n_days < 3:
        raise ValueError(f"数据中仅有 {n_days} 个不同日期，至少需要 3 天才能切分 train/valid/test。")

    # -------------------------
    # 先找 test：从最后一天往前扩，直到满足正样本约束
    # -------------------------
    test_start_idx = None
    test_click_pos = 0
    test_conv_pos = 0

    for i in range(n_days - 1, -1, -1):
        cand_test_dates = sorted_dates[i:]
        ok, click_pos, conv_pos = has_required_positive(
            cand_test_dates,
            date_action_counter,
            require_click_positive=require_click_positive,
            require_conversion_positive=require_conversion_positive
        )
        if ok:
            test_start_idx = i
            test_click_pos = click_pos
            test_conv_pos = conv_pos
            break

    if test_start_idx is None:
        raise ValueError("无法构造 test：最后时间段中不存在满足条件的正样本。")

    # 至少还要给 train 和 valid 留 1 天
    if test_start_idx < 2:
        raise ValueError("无法构造合法切分：test 为最短合法尾段后，前面不足以再划分 train 和 valid。")

    # -------------------------
    # 再找 valid：必须紧挨 test，且为最短合法时间段
    # valid = sorted_dates[j:test_start_idx]
    # 从 test_start_idx-1 往前扩
    # -------------------------
    valid_start_idx = None
    valid_click_pos = 0
    valid_conv_pos = 0

    for j in range(test_start_idx - 1, -1, -1):
        cand_valid_dates = sorted_dates[j:test_start_idx]
        ok, click_pos, conv_pos = has_required_positive(
            cand_valid_dates,
            date_action_counter,
            require_click_positive=require_click_positive,
            require_conversion_positive=require_conversion_positive
        )
        if ok:
            valid_start_idx = j
            valid_click_pos = click_pos
            valid_conv_pos = conv_pos
            break

    if valid_start_idx is None:
        raise ValueError("无法构造 valid：test 前面的紧邻时间段中不存在满足条件的正样本。")

    # train 至少留 1 天
    if valid_start_idx < 1:
        raise ValueError("无法构造合法切分：valid 为最短合法段后，前面不足以再留给 train。")

    train_dates = sorted_dates[:valid_start_idx]
    valid_dates = sorted_dates[valid_start_idx:test_start_idx]
    test_dates = sorted_dates[test_start_idx:]

    return {
        "n_days": n_days,
        "n_train": len(train_dates),
        "n_valid": len(valid_dates),
        "n_test": len(test_dates),
        "train_dates": train_dates,
        "valid_dates": valid_dates,
        "test_dates": test_dates,
        "valid_start_date": valid_dates[0],
        "test_start_date": test_dates[0],
        "valid_click_pos": int(valid_click_pos),
        "valid_conv_pos": int(valid_conv_pos),
        "test_click_pos": int(test_click_pos),
        "test_conv_pos": int(test_conv_pos),
    }


# ================================================================
# 3. 加载 user/item 特征
# ================================================================

def load_user_features(data_dir: Path) -> pd.DataFrame:
    fp = data_dir / "user_feat.parquet"
    print(f"  [Load] {fp.name}")
    uf = pd.read_parquet(fp)
    uf = ensure_str_columns(uf)

    if "user_id" not in uf.columns:
        raise ValueError("user_feat.parquet 缺少 user_id 列")

    out = pd.DataFrame()
    out["user_id"] = pd.to_numeric(uf["user_id"], errors="coerce").fillna(-1).astype(np.int64)

    for col in USER_SCALAR_FEATURES:
        if col in uf.columns:
            out[col] = pd.to_numeric(uf[col], errors="coerce").fillna(0).astype(np.int32)
        else:
            out[col] = np.zeros(len(uf), dtype=np.int32)

    for col in USER_LIST_FEATURES:
        if col in uf.columns:
            out[col] = uf[col].map(lambda x: normalize_list_feature(x, USER_LIST_KEEP))
        else:
            out[col] = [[0] * USER_LIST_KEEP for _ in range(len(uf))]

    out = out[out["user_id"] >= 0].drop_duplicates(subset=["user_id"], keep="last").reset_index(drop=True)
    print(f"         {len(out):,} users")
    return out


def load_item_features(data_dir: Path) -> pd.DataFrame:
    fp = data_dir / "item_feat.parquet"
    print(f"  [Load] {fp.name}")
    itf = pd.read_parquet(fp)
    itf = ensure_str_columns(itf)

    if "item_id" not in itf.columns:
        raise ValueError("item_feat.parquet 缺少 item_id 列")

    out = pd.DataFrame()
    out["item_id"] = pd.to_numeric(itf["item_id"], errors="coerce").fillna(-1).astype(np.int64)

    for col in ITEM_STATIC_FEATURES:
        if col in itf.columns:
            out[col] = pd.to_numeric(itf[col], errors="coerce").fillna(0).astype(np.int32)
        else:
            out[col] = np.zeros(len(itf), dtype=np.int32)

    out = out[out["item_id"] >= 0].drop_duplicates(subset=["item_id"], keep="last").reset_index(drop=True)
    print(f"         {len(out):,} items")
    return out


# ================================================================
# 4. feature size / 准备 user features
# ================================================================

def build_feature_size_meta(user_feat_df: pd.DataFrame, item_feat_df: pd.DataFrame) -> dict:
    feat_size = {}

    for col in USER_SCALAR_FEATURES:
        feat_size[col] = get_scalar_vocab_size(user_feat_df[col])

    for col in USER_LIST_FEATURES:
        feat_size[col] = get_list_vocab_size(user_feat_df[col])

    # 新增：把 item static features 显式加入 vocab_size 统计
    for col in ITEM_STATIC_FEATURES:
        feat_size[col] = get_scalar_vocab_size(item_feat_df[col])

    feat_size["day_of_week"] = 8
    feat_size["is_weekend"] = 3
    feat_size["hour"] = 25
    return feat_size


def prepare_user_features(user_feat_df: pd.DataFrame) -> pd.DataFrame:
    out = user_feat_df[["user_id"]].copy()

    for col in USER_SCALAR_FEATURES:
        out[col] = pd.to_numeric(user_feat_df[col], errors="coerce").fillna(0).astype(np.int32)

    for col in USER_LIST_FEATURES:
        out[col] = user_feat_df[col].map(lambda x: normalize_list_feature(x, USER_LIST_KEEP))

    return out


# ================================================================
# 5. seq.parquet 分批读取 / 预处理
# ================================================================

def iter_seq_batches(seq_fp: Path, batch_rows: int = 50000):
    pf = pq.ParquetFile(seq_fp)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        df = table.to_pandas()
        df = ensure_str_columns(df)

        n = len(df)
        if n <= batch_rows:
            yield df
        else:
            for start in range(0, n, batch_rows):
                yield df.iloc[start:start + batch_rows].copy()


def preprocess_seq_batch(batch_df: pd.DataFrame, timestamp_unit: str) -> pd.DataFrame:
    if "user_id" not in batch_df.columns or "seq" not in batch_df.columns:
        raise ValueError("seq.parquet 需要包含 user_id 和 seq 两列")

    batch_df = batch_df[["user_id", "seq"]].copy()
    batch_df["user_id"] = pd.to_numeric(batch_df["user_id"], errors="coerce")
    batch_df = batch_df.dropna(subset=["user_id"])
    batch_df["user_id"] = batch_df["user_id"].astype(np.int64)

    exploded = batch_df.explode("seq", ignore_index=True)
    exploded = exploded[exploded["seq"].notna()].copy()

    if len(exploded) == 0:
        return pd.DataFrame(columns=[
            "user_id", "item_id", "action_type", "timestamp",
            "date", "day_of_week", "is_weekend", "hour"
        ])

    event_df = pd.json_normalize(exploded["seq"])
    event_df.columns = [str(c) for c in event_df.columns]

    df = pd.concat(
        [
            exploded[["user_id"]].reset_index(drop=True),
            event_df.reset_index(drop=True),
        ],
        axis=1,
    )

    required = ["item_id", "action_type", "timestamp"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"展开后的 seq 事件缺少字段: {c}")

    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    df["action_type"] = pd.to_numeric(df["action_type"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df.dropna(subset=["user_id", "item_id", "action_type", "timestamp"], inplace=True)

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "user_id", "item_id", "action_type", "timestamp",
            "date", "day_of_week", "is_weekend", "hour"
        ])

    df["user_id"] = df["user_id"].astype(np.int64)
    df["item_id"] = df["item_id"].astype(np.int64)
    df["action_type"] = df["action_type"].astype(np.int8)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    dt = pd.to_datetime(df["timestamp"], unit=timestamp_unit, utc=True, errors="coerce").dt.tz_convert(BEIJING_TZ)
    valid_mask = dt.notna()
    df = df.loc[valid_mask].copy()
    dt = dt.loc[valid_mask]

    raw_day_of_week = dt.dt.dayofweek.astype(np.int8)
    raw_is_weekend = (dt.dt.dayofweek >= 5).astype(np.int8)
    raw_hour = dt.dt.hour.astype(np.int8)

    df["date"] = (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).astype(np.int32)
    df["day_of_week"] = (raw_day_of_week + 1).astype(np.int32)
    df["is_weekend"] = (raw_is_weekend + 1).astype(np.int32)
    df["hour"] = (raw_hour + 1).astype(np.int32)

    return df[[
        "user_id", "item_id", "action_type", "timestamp",
        "date", "day_of_week", "is_weekend", "hour"
    ]]


# ================================================================
# 6. Phase 1: 先按 user_id hash 分区
# ================================================================

def phase1_partition_seq_to_parquet(
    data_dir: Path,
    tmp_dir: Path,
    n_parts: int,
    seq_batch_rows: int,
    buffer_flush_size: int,
):
    seq_fp = data_dir / "seq.parquet"
    if not seq_fp.exists():
        raise FileNotFoundError(f"找不到文件: {seq_fp}")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in range(n_parts):
        (tmp_dir / f"part_{p:03d}").mkdir(parents=True, exist_ok=True)

    sample_user_rows = next(iter_seq_batches(seq_fp, batch_rows=1000))
    sample_exp = sample_user_rows[["seq"]].explode("seq", ignore_index=True)
    sample_exp = sample_exp[sample_exp["seq"].notna()]
    sample_evt = pd.json_normalize(sample_exp["seq"]) if len(sample_exp) > 0 else pd.DataFrame()
    sample_timestamps = sample_evt["timestamp"].tolist()[:100] if "timestamp" in sample_evt.columns else []
    timestamp_unit = detect_timestamp_unit_from_sample(sample_timestamps)

    print(f"  [Phase1] 检测 timestamp 单位: {timestamp_unit}")
    print(f"  [Phase1] 时间特征按北京时间 {BEIJING_TZ} 提取，并按 KuaiRand 风格编码")

    user_counts = defaultdict(int)
    item_ids = set()
    all_dates = set()
    raw_action_type_counter = defaultdict(int)
    date_action_counter = defaultdict(lambda: defaultdict(int))
    user_action_counter = defaultdict(lambda: defaultdict(int))

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

    total_rows = 0
    total_users = 0

    for batch_idx, batch_df in enumerate(iter_seq_batches(seq_fp, batch_rows=seq_batch_rows), start=1):
        total_users += len(batch_df)
        chunk = preprocess_seq_batch(batch_df, timestamp_unit=timestamp_unit)

        if len(chunk) == 0:
            del batch_df, chunk
            gc.collect()
            continue

        vc = chunk["user_id"].value_counts(sort=False)
        for uid, cnt in zip(vc.index, vc.values):
            user_counts[int(uid)] += int(cnt)

        item_ids.update(chunk["item_id"].unique().tolist())
        all_dates.update(chunk["date"].unique().tolist())

        ac = chunk["action_type"].value_counts(sort=False)
        for a, cnt in zip(ac.index, ac.values):
            raw_action_type_counter[int(a)] += int(cnt)

        dac = chunk.groupby(["date", "action_type"]).size()
        for (d, a), cnt in dac.items():
            date_action_counter[int(d)][int(a)] += int(cnt)

        uac = chunk.groupby(["user_id", "action_type"]).size()
        for (uid, a), cnt in uac.items():
            user_action_counter[int(uid)][int(a)] += int(cnt)

        part_arr = chunk["user_id"].values.astype(np.int64) % n_parts
        for pid in range(n_parts):
            mask = part_arr == pid
            n_match = int(mask.sum())
            if n_match > 0:
                buffers[pid].append(chunk.loc[mask].copy())
                buf_sizes[pid] += n_match
                if buf_sizes[pid] >= buffer_flush_size:
                    flush(pid)

        total_rows += len(chunk)

        if batch_idx % 10 == 0:
            print(f"  [Phase1] batch={batch_idx:5d}  users={total_users:,}  interactions={total_rows:,}")

        del batch_df, chunk, vc, ac, dac, uac, part_arr
        gc.collect()

    for pid in range(n_parts):
        flush(pid)

    positive_users, kept_users_n, dropped_all_negative_users_n = count_positive_users_from_action_counter(user_action_counter)

    print(
        f"  [Phase1] 完成: {total_rows:,} interactions, "
        f"{len(user_counts):,} users, {len(item_ids):,} items, "
        f"{len(all_dates)} dates"
    )
    print(
        f"  [Phase1] 全负样本用户过滤统计: "
        f"kept_positive_users={kept_users_n:,}, "
        f"dropped_all_negative_users={dropped_all_negative_users_n:,}"
    )

    return {
        "user_counts": dict(user_counts),
        "item_ids": item_ids,
        "all_dates": all_dates,
        "raw_action_type_counter": dict(raw_action_type_counter),
        "date_action_counter": {int(d): {int(a): int(c) for a, c in mp.items()} for d, mp in date_action_counter.items()},
        "positive_users": positive_users,
        "timestamp_unit": timestamp_unit,
        "total_rows": total_rows,
        "dropped_all_negative_users": int(dropped_all_negative_users_n),
    }


# ================================================================
# 7. 处理单个 partition，得到“全局 id 版本”的 split 数据
# ================================================================

def process_partition(
    tmp_dir: Path,
    pid: int,
    valid_users: set,
    global_user_index_map: dict,
    global_item_id_map: dict,
    raw2action_name: dict,
    action_name2code: dict,
    user_feat_ready: pd.DataFrame,
    valid_start_date: int,
    test_start_date: int,
):
    part_dir = tmp_dir / f"part_{pid:03d}"
    files = sorted(part_dir.glob("*.parquet"))
    if not files:
        return None

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

    df = df.merge(user_feat_ready, on="user_id", how="left")

    for col in USER_SCALAR_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int32)

    for col in USER_LIST_FEATURES:
        if col not in df.columns:
            df[col] = [[0] * USER_LIST_KEEP for _ in range(len(df))]
        df[col] = df[col].map(lambda x: normalize_list_feature(x, USER_LIST_KEEP))

    df.sort_values(["user_id", "timestamp", "item_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["global_user_index"] = df["user_id"].map(global_user_index_map).astype(np.int32)
    df["global_item_id"] = df["item_id"].map(global_item_id_map).fillna(0).astype(np.int32)

    df["user_id"] = (df["global_user_index"] + 1).astype(np.int32)

    df["is_click"] = (df["action_type"] == 1).astype(np.float32)
    df["is_conversion"] = (df["action_type"] == 2).astype(np.float32)

    df["action_name"] = df["action_type"].map(raw2action_name)
    df["action"] = df["action_name"].map(action_name2code).fillna(0).astype(np.int32)

    for col in CONTEXT_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int32)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    df["seq_len"] = df.groupby("global_user_index", sort=False).cumcount().astype(np.int32)

    user_info_rows = []
    for g_uidx, gdf in df.groupby("global_user_index", sort=True):
        user_info_rows.append(
            {
                "user_index": int(g_uidx),
                "full_item_seq": gdf["global_item_id"].astype(int).tolist(),
                "full_action_seq": gdf["action"].astype(int).tolist(),
                "full_timestamp_seq": gdf["timestamp"].astype(np.int64).tolist(),
            }
        )
    user_info_df = pd.DataFrame(user_info_rows)

    date_col = df["date"]
    train_df = df[date_col < valid_start_date].copy()
    valid_df = df[(date_col >= valid_start_date) & (date_col < test_start_date)].copy()
    test_df = df[date_col >= test_start_date].copy()

    def _select_final(sdf):
        if len(sdf) == 0:
            return pd.DataFrame(columns=FINAL_COLUMNS)

        out = sdf.copy()
        out["user_index"] = out["global_user_index"].astype(np.int32)
        out["item_index"] = out["global_item_id"].astype(np.int32)

        present = [c for c in FINAL_COLUMNS if c in out.columns]
        return out[present].reset_index(drop=True)

    result = {
        "train": _select_final(train_df),
        "valid": _select_final(valid_df),
        "test": _select_final(test_df),
        "user_info": user_info_df,
    }

    del df, train_df, valid_df, test_df, user_info_df
    gc.collect()
    return result


# ================================================================
# 8. split block 管理器
# ================================================================

class SplitBlockManager:
    def __init__(self, split_name: str, root_dir: Path, num_blocks: int):
        self.split_name = split_name
        self.root_dir = root_dir / split_name
        self.data_dir = self.root_dir / "data"
        self.user_info_dir = self.root_dir / "user_info"
        self.item_info_dir = self.root_dir / "item_info"

        self.num_blocks = int(num_blocks)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.user_info_dir.mkdir(parents=True, exist_ok=True)
        self.item_info_dir.mkdir(parents=True, exist_ok=True)

        self.data_writers = [None] * self.num_blocks
        self.user_writers = [None] * self.num_blocks

        self.block_rows = [0] * self.num_blocks
        self.block_user_rows = [0] * self.num_blocks

        self.partition_to_block = {}

        self.user_maps = [dict() for _ in range(self.num_blocks)]
        self.item_maps = [{0: 0} for _ in range(self.num_blocks)]

        self.next_user_local = [0] * self.num_blocks
        self.next_item_local = [1] * self.num_blocks

    def choose_block(self, n_rows: int) -> int:
        return int(np.argmin(self.block_rows))

    def _get_or_add_user_local(self, bid: int, global_user_index: int) -> int:
        mp = self.user_maps[bid]
        if global_user_index not in mp:
            mp[global_user_index] = self.next_user_local[bid]
            self.next_user_local[bid] += 1
        return mp[global_user_index]

    def _get_or_add_item_local(self, bid: int, global_item_id: int) -> int:
        mp = self.item_maps[bid]
        if global_item_id not in mp:
            mp[global_item_id] = self.next_item_local[bid]
            self.next_item_local[bid] += 1
        return mp[global_item_id]

    def _write_table(self, writer_list, bid: int, out_fp: Path, df: pd.DataFrame):
        if len(df) == 0:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer_list[bid] is None:
            writer_list[bid] = pq.ParquetWriter(str(out_fp), table.schema)
        writer_list[bid].write_table(table)

    def append_partition(self, pid: int, split_df: pd.DataFrame, user_info_df: pd.DataFrame):
        if len(split_df) == 0:
            return None

        bid = self.choose_block(len(split_df))
        self.partition_to_block[int(pid)] = int(bid)

        active_users = pd.unique(split_df["user_index"].astype(np.int64))
        active_users_set = set(active_users.tolist())

        sub_ui = user_info_df[user_info_df["user_index"].isin(active_users_set)].copy()
        if len(sub_ui) == 0:
            raise ValueError(
                f"[{self.split_name}] partition={pid} 有样本，但找不到对应 user_info，逻辑异常。"
            )

        out_ui_rows = []
        for row in sub_ui.itertuples(index=False):
            g_user = int(row.user_index)
            l_user = self._get_or_add_user_local(bid, g_user)

            g_item_seq = row.full_item_seq if isinstance(row.full_item_seq, (list, tuple, np.ndarray)) else []
            l_item_seq = [self._get_or_add_item_local(bid, safe_int(x, 0)) for x in g_item_seq]

            g_action_seq = row.full_action_seq if isinstance(row.full_action_seq, (list, tuple, np.ndarray)) else []
            g_time_seq = row.full_timestamp_seq if isinstance(row.full_timestamp_seq, (list, tuple, np.ndarray)) else []

            out_ui_rows.append(
                {
                    "user_index": np.int32(l_user),
                    "full_item_seq": [int(x) for x in l_item_seq],
                    "full_action_seq": [int(x) for x in g_action_seq],
                    "full_timestamp_seq": [int(x) for x in g_time_seq],
                }
            )

        out_ui_df = pd.DataFrame(out_ui_rows)

        user_map = self.user_maps[bid]
        item_map = self.item_maps[bid]

        out_df = split_df.copy()
        out_df["user_index"] = out_df["user_index"].map(user_map).astype(np.int32)
        out_df["item_index"] = out_df["item_index"].map(item_map).astype(np.int32)

        data_fp = self.data_dir / f"part-{bid:05d}.parquet"
        ui_fp = self.user_info_dir / f"part-{bid:05d}.parquet"

        self._write_table(self.data_writers, bid, data_fp, out_df)
        self._write_table(self.user_writers, bid, ui_fp, out_ui_df)

        self.block_rows[bid] += len(out_df)
        self.block_user_rows[bid] += len(out_ui_df)

        del out_df, out_ui_df, sub_ui, out_ui_rows
        gc.collect()
        return bid

    def close_writers(self):
        for w in self.data_writers:
            if w is not None:
                w.close()
        for w in self.user_writers:
            if w is not None:
                w.close()

    def write_item_info_blocks(self, global_item_lookup: pd.DataFrame):
        for bid in range(self.num_blocks):
            if self.block_rows[bid] == 0:
                continue

            item_map = self.item_maps[bid]
            local_size = max(item_map.values()) if len(item_map) > 0 else 0

            inv_global_item_id = np.zeros(local_size + 1, dtype=np.int32)
            for g_item_id, l_item_idx in item_map.items():
                inv_global_item_id[l_item_idx] = np.int32(g_item_id)

            out = {
                "item_index": np.arange(local_size + 1, dtype=np.int32),
                "item_id": inv_global_item_id.astype(np.int32),
            }

            if local_size > 0:
                gids = inv_global_item_id[1:].astype(np.int32)
                feat_df = global_item_lookup.reindex(gids).fillna(0)

                for col in ITEM_STATIC_FEATURES:
                    arr = np.zeros(local_size + 1, dtype=np.int32)
                    arr[1:] = feat_df[col].to_numpy(dtype=np.int32, copy=False)
                    out[col] = arr
            else:
                for col in ITEM_STATIC_FEATURES:
                    out[col] = np.zeros(1, dtype=np.int32)

            item_info_df = pd.DataFrame(out)
            out_fp = self.item_info_dir / f"part-{bid:05d}.parquet"
            item_info_df.to_parquet(out_fp, index=False, engine="pyarrow")

            del item_info_df
            gc.collect()

    def build_manifest(self):
        blocks = []
        for bid in range(self.num_blocks):
            if self.block_rows[bid] == 0:
                continue
            blocks.append(
                {
                    "block_id": bid,
                    "rows": int(self.block_rows[bid]),
                    "users": int(len(self.user_maps[bid])),
                    "items": int(len(self.item_maps[bid]) - 1),
                    "data_file": str(self.data_dir / f"part-{bid:05d}.parquet"),
                    "user_info_file": str(self.user_info_dir / f"part-{bid:05d}.parquet"),
                    "item_info_file": str(self.item_info_dir / f"part-{bid:05d}.parquet"),
                    "source_partitions": [
                        int(pid) for pid, b in self.partition_to_block.items() if b == bid
                    ],
                }
            )
        return {
            "split": self.split_name,
            "num_blocks_configured": int(self.num_blocks),
            "num_blocks_written": int(len(blocks)),
            "blocks": blocks,
        }


# ================================================================
# 9. 构建全局 item lookup（供各 block item_info 使用）
# ================================================================

def build_global_item_lookup(item_feat_df: pd.DataFrame, global_item_id_map: dict) -> pd.DataFrame:
    vf = item_feat_df[item_feat_df["item_id"].isin(global_item_id_map)].copy()
    vf["global_item_id"] = vf["item_id"].map(global_item_id_map).astype(np.int32)
    vf = vf[["global_item_id"] + ITEM_STATIC_FEATURES].drop_duplicates(subset=["global_item_id"], keep="last")
    vf = vf.set_index("global_item_id").sort_index()

    for col in ITEM_STATIC_FEATURES:
        vf[col] = pd.to_numeric(vf[col], errors="coerce").fillna(0).astype(np.int32)

    return vf


# ================================================================
# 10. 主流程
# ================================================================

def preprocess_and_split_blocked(
    data_dir: str = "./",
    output_dir: str = "../TencentGR_10M_Action_Blocked",
    min_user_interactions: int = 10,
    n_user_parts: int = 32,
    seq_batch_rows: int = 50000,
    buffer_flush_size: int = 500000,
    train_blocks: int = 16,
    valid_blocks: int = 8,
    test_blocks: int = 8,
    overwrite: bool = False,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    prepare_output_dir(output_dir, overwrite=overwrite)
    tmp_dir = output_dir / "_tmp_partitions"

    print("\n[Step 1/8] 加载 user/item 特征 & 构建 feature size")
    user_feat_df_raw = load_user_features(data_dir)
    item_feat_df = load_item_features(data_dir)

    feature_size_meta = build_feature_size_meta(user_feat_df_raw, item_feat_df)
    raw2action_name, action_name2code = build_action_maps()
    user_feat_ready = prepare_user_features(user_feat_df_raw)

    print(f"  action 种类: {len(action_name2code)}")
    print("  用户特征准备完成：标量保留原始 id，list 保留单列 list[int]\n")

    print("[Step 2/8] Phase 1: seq.parquet -> 按 user_id hash 分区 parquet")
    phase1_stat = phase1_partition_seq_to_parquet(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        n_parts=n_user_parts,
        seq_batch_rows=seq_batch_rows,
        buffer_flush_size=buffer_flush_size,
    )

    user_counts = phase1_stat["user_counts"]
    item_ids = phase1_stat["item_ids"]
    all_dates = phase1_stat["all_dates"]
    raw_action_type_counter = phase1_stat["raw_action_type_counter"]
    total_rows_phase1 = phase1_stat["total_rows"]
    timestamp_unit = phase1_stat["timestamp_unit"]
    date_action_counter = phase1_stat["date_action_counter"]
    positive_users = phase1_stat["positive_users"]
    dropped_all_negative_users = phase1_stat["dropped_all_negative_users"]
    print("")

    print("[Step 2.5/8] 按固定规则切分：最短合法 test + 最短合法 valid + 剩余 train")
    sorted_dates = sorted(all_dates)
    split_info = build_minimal_tail_splits(
        sorted_dates=sorted_dates,
        date_action_counter=date_action_counter,
        require_click_positive=True,
        require_conversion_positive=True,
    )

    train_dates = split_info["train_dates"]
    valid_dates = split_info["valid_dates"]
    test_dates = split_info["test_dates"]
    valid_start_date = split_info["valid_start_date"]
    test_start_date = split_info["test_start_date"]

    print(f"  总天数:    {split_info['n_days']} 天 ({fmt_date_int(sorted_dates[0])} ~ {fmt_date_int(sorted_dates[-1])})")
    print(f"  训练集:    {split_info['n_train']} 天  {fmt_date_int(train_dates[0])} ~ {fmt_date_int(train_dates[-1])}")
    print(f"  验证集:    {split_info['n_valid']} 天  {fmt_date_int(valid_dates[0])} ~ {fmt_date_int(valid_dates[-1])}")
    print(f"  测试集:    {split_info['n_test']} 天  {fmt_date_int(test_dates[0])} ~ {fmt_date_int(test_dates[-1])}")
    print(f"  valid 正样本: click={split_info['valid_click_pos']:,}, conversion={split_info['valid_conv_pos']:,}")
    print(f"  test  正样本: click={split_info['test_click_pos']:,}, conversion={split_info['test_conv_pos']:,}")
    print("")

    print("[Step 3/8] 过滤低频用户 + 过滤全负样本用户 + 构建全局 ID")
    valid_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    n_dropped_low_freq = len(user_counts) - len(valid_users)

    valid_users = valid_users.intersection(positive_users)
    n_after_positive_filter = len(valid_users)

    print(f"  低频过滤后有效用户: {len(user_counts) - n_dropped_low_freq:,}")
    print(f"  全负样本用户过滤数: {dropped_all_negative_users:,}")
    print(f"  最终保留用户数:     {n_after_positive_filter:,}")

    sorted_users = sorted(valid_users)
    global_user_index_map = {u: i for i, u in enumerate(sorted_users)}
    sorted_items = sorted(item_ids)
    global_item_id_map = {it: i + 1 for i, it in enumerate(sorted_items)}

    print(f"  全局物品数: {len(global_item_id_map):,}")

    vocab_size = {
        "user_id": len(global_user_index_map) + 1,
        "item_id": len(global_item_id_map) + 1,
        "timestamp": 0,
        "action": len(action_name2code) + 1,
    }
    for col in USER_SCALAR_FEATURES:
        vocab_size[col] = int(feature_size_meta[col])
    for col in USER_LIST_FEATURES:
        vocab_size[col] = int(feature_size_meta[col])
    for col in ITEM_STATIC_FEATURES:
        vocab_size[col] = int(feature_size_meta[col])
    for col in CONTEXT_FEATURES:
        vocab_size[col] = int(feature_size_meta[col])

    print("")

    print("[Step 4/8] 逐 partition 编码 + 切分 + 直接写 block data/user_info")

    managers = {
        "train": SplitBlockManager("train", output_dir, train_blocks),
        "valid": SplitBlockManager("valid", output_dir, valid_blocks),
        "test": SplitBlockManager("test", output_dir, test_blocks),
    }

    sample_counts = {"train": 0, "valid": 0, "test": 0}
    max_seq = 0

    for pid in range(n_user_parts):
        result = process_partition(
            tmp_dir=tmp_dir,
            pid=pid,
            valid_users=valid_users,
            global_user_index_map=global_user_index_map,
            global_item_id_map=global_item_id_map,
            raw2action_name=raw2action_name,
            action_name2code=action_name2code,
            user_feat_ready=user_feat_ready,
            valid_start_date=valid_start_date,
            test_start_date=test_start_date,
        )
        if result is None:
            continue

        user_info_df = result["user_info"]
        if len(user_info_df) > 0:
            part_max_seq = user_info_df["full_item_seq"].map(len).max()
            if int(part_max_seq) > max_seq:
                max_seq = int(part_max_seq)

        for split_name in ["train", "valid", "test"]:
            sdf = result[split_name]
            if len(sdf) == 0:
                continue
            managers[split_name].append_partition(pid=pid, split_df=sdf, user_info_df=user_info_df)
            sample_counts[split_name] += len(sdf)

        done = sum(sample_counts.values())
        print(f"  partition {pid + 1:3d}/{n_user_parts}: 累计写出 {done:,} 行")

        del result, user_info_df
        gc.collect()

    for mgr in managers.values():
        mgr.close_writers()

    total = sum(sample_counts.values())
    print(
        f"\n  train={sample_counts['train']:,}  "
        f"valid={sample_counts['valid']:,}  "
        f"test={sample_counts['test']:,}"
    )
    print(f"  总计={total:,}\n")

    if total == 0:
        raise ValueError("处理后数据为空，请检查输入数据或调小 min_user_interactions。")

    print("[Step 5/8] 为每个 block 构建 item_info")
    global_item_lookup = build_global_item_lookup(item_feat_df, global_item_id_map)

    for split_name in ["train", "valid", "test"]:
        print(f"  [Build item_info] {split_name}")
        managers[split_name].write_item_info_blocks(global_item_lookup)
        gc.collect()

    print("\n[Step 6/8] 保存 block_manifest.json")
    block_manifest = {
        "train": managers["train"].build_manifest(),
        "valid": managers["valid"].build_manifest(),
        "test": managers["test"].build_manifest(),
    }
    with open(output_dir / "block_manifest.json", "w", encoding="utf-8") as f:
        json.dump(block_manifest, f, ensure_ascii=False, indent=4)
    print("  [Saved] block_manifest.json")

    print("\n[Step 7/8] 保存 meta_data.json")
    meta = {
        "sample_size": {
            "total": int(total),
            "train": int(sample_counts["train"]),
            "valid": int(sample_counts["valid"]),
            "test": int(sample_counts["test"]),
            "phase1_total_interactions": int(total_rows_phase1),
        },
        "split_by_minimal_tail_with_positive_constraints": {
            "timezone": BEIJING_TZ,
            "timestamp_unit": timestamp_unit,
            "train_days": split_info["n_train"],
            "train_range": f"{fmt_date_int(train_dates[0])} ~ {fmt_date_int(train_dates[-1])}",
            "valid_days": split_info["n_valid"],
            "valid_range": f"{fmt_date_int(valid_dates[0])} ~ {fmt_date_int(valid_dates[-1])}",
            "test_days": split_info["n_test"],
            "test_range": f"{fmt_date_int(test_dates[0])} ~ {fmt_date_int(test_dates[-1])}",
            "valid_click_pos": int(split_info["valid_click_pos"]),
            "valid_conversion_pos": int(split_info["valid_conv_pos"]),
            "test_click_pos": int(split_info["test_click_pos"]),
            "test_conversion_pos": int(split_info["test_conv_pos"]),
            "rule": "test=最后最短合法正样本时间段; valid=紧邻test的最短合法正样本时间段; 其余=train",
        },
        "user_filtering": {
            "min_user_interactions": int(min_user_interactions),
            "dropped_all_negative_users": int(dropped_all_negative_users),
            "rule": "用户完整行为序列若既无 click 也无 conversion，则过滤",
        },
        "blocked_layout": {
            "train_blocks": int(train_blocks),
            "valid_blocks": int(valid_blocks),
            "test_blocks": int(test_blocks),
            "train": {
                "data_dir": "train/data",
                "user_info_dir": "train/user_info",
                "item_info_dir": "train/item_info",
            },
            "valid": {
                "data_dir": "valid/data",
                "user_info_dir": "valid/user_info",
                "item_info_dir": "valid/item_info",
            },
            "test": {
                "data_dir": "test/data",
                "user_info_dir": "test/user_info",
                "item_info_dir": "test/item_info",
            },
            "block_pair_rule": (
                "同一 split 下，data/user_info/item_info 使用相同的 part-xxxxx 编号配对读取。"
            ),
            "local_index_rule": {
                "user_index": "block-local dense index, starts from 0",
                "item_index": "block-local dense index, 0 reserved for padding",
                "user_id": "global feature id, consistent across blocks",
                "item_id": "global feature id, consistent across blocks",
            },
        },
        "vocab_size": {k: int(v) for k, v in vocab_size.items()},
        "label": LABEL_COLUMNS,
        "action_vocab": {k: int(v) for k, v in action_name2code.items()},
        "action_vocab_desc": (
            "编码后的 action 词表，用于 dataloader 基于 full_action_seq "
            "构造 task-specific token masks。"
        ),
        "action_mapping": {
            "raw_action_type_to_name": {str(k): v for k, v in raw2action_name.items()},
            "action_name_to_code": {k: int(v) for k, v in action_name2code.items()},
        },
        "user_info_schema": {
            "fields": [
                "user_index",
                "full_item_seq",
                "full_action_seq",
                "full_timestamp_seq",
            ],
            "desc": (
                "这里的 user_index / full_item_seq 中的 item index 都是 block-local index；"
                "full_action_seq / full_timestamp_seq 为全局时间顺序序列。"
            ),
        },
        "item_info_schema": {
            "fields": [
                "item_index",
                "item_id",
            ] + ITEM_STATIC_FEATURES,
            "desc": (
                "item_index 为 block-local index；item_id 为全局 item feature id，"
                "用于 embedding 一致性。"
            ),
        },
        "feature_schema": {
            "user_scalar_features": USER_SCALAR_FEATURES,
            "user_list_features": USER_LIST_FEATURES,
            "user_list_keep": USER_LIST_KEEP,
            "user_list_storage": "single-column list[int] with fixed length",
            "user_scalar_encoding": "keep raw integer ids; vocab_size=max+1",
            "user_list_encoding": "keep raw list[int]; vocab_size=max_sub_feature_id+1",
            "item_static_features": ITEM_STATIC_FEATURES,
            "item_static_encoding": "keep raw integer ids; vocab_size=max+1",
            "context_features": CONTEXT_FEATURES,
            "time_feature_encoding": {
                "day_of_week": "0~6 -> 1~7, 0 reserved for padding/unknown",
                "is_weekend": "0/1 -> 1~2, 0 reserved for padding/unknown",
                "hour": "0~23 -> 1~24, 0 reserved for padding/unknown",
            },
            "timestamp_kept_raw": True,
            "context_time_timezone": BEIJING_TZ,
        },
        "max_len": {
            "full_item_seq": int(max_seq),
            "full_action_seq": int(max_seq),
            "full_timestamp_seq": int(max_seq),
        },
        "raw_action_type_counter": {
            str(k): int(v) for k, v in sorted(raw_action_type_counter.items())
        },
    }

    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    print("  [Saved] meta_data.json")

    print("\n[Step 8/8] 清理临时分区文件")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  Done.")

    print("\n" + "=" * 68)
    print("  TencentGR / TAAC2025 Blocked Seq-Action Preprocess Done")
    print("=" * 68)
    print(f"输出目录: {output_dir}\n")
    print("目录结构示例：")
    print(f"  {output_dir / 'train' / 'data'}")
    print(f"  {output_dir / 'train' / 'user_info'}")
    print(f"  {output_dir / 'train' / 'item_info'}")
    print(f"  {output_dir / 'valid' / 'data'}")
    print(f"  {output_dir / 'test' / 'data'}")
    print(f"  {output_dir / 'meta_data.json'}")
    print(f"  {output_dir / 'block_manifest.json'}")


# ================================================================
# 11. CLI
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess TencentGR/TAAC2025 seq-action data directly into blocked train/valid/test with block-local user_info/item_info."
    )
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--output_dir", type=str, default="../TencentGR_10M_Action_Blocked")

    parser.add_argument("--min_user_interactions", type=int, default=10)

    parser.add_argument("--n_user_parts", type=int, default=20,
                        help="Phase1 按 user hash 的临时分区数，建议 >= max(train_blocks, valid_blocks, test_blocks)")
    parser.add_argument("--seq_batch_rows", type=int, default=50000,
                        help="读取 seq.parquet 时单次处理多少行 user-record")
    parser.add_argument("--buffer_flush_size", type=int, default=500000,
                        help="临时分区缓存多少 interaction 后落盘")

    parser.add_argument("--train_blocks", type=int, default=16)
    parser.add_argument("--valid_blocks", type=int, default=8)
    parser.add_argument("--test_blocks", type=int, default=8)

    parser.add_argument("--overwrite", action="store_true",  default=True,
                        help="若输出目录已存在，则删除后重建")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    preprocess_and_split_blocked(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_user_interactions=args.min_user_interactions,
        n_user_parts=args.n_user_parts,
        seq_batch_rows=args.seq_batch_rows,
        buffer_flush_size=args.buffer_flush_size,
        train_blocks=args.train_blocks,
        valid_blocks=args.valid_blocks,
        test_blocks=args.test_blocks,
        overwrite=args.overwrite,
    )