#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_Kuairand_seq_action.py — KuaiRand-27K (内存优化版, 按日期比例切分)
===========================================================================
通过分批处理避免内存溢出 (OOM)。

切分策略（按日期 8:1:1）:
  - 将所有日期排序后，按天数比例 8:1:1 划分
  - 前 ~80% 天 → 训练集
  - 中间 ~10% 天 → 验证集
  - 最后 ~10% 天 → 测试集

优化策略:
  Phase 1 — CSV → 按用户哈希分区的 Parquet 中间文件
  Phase 2 — 逐分区编码 + 按日期切分 + 增量写出
  Phase 3 — 保存 user_info / item_info / meta_data + 清理临时文件

本版本说明:
  - 不再构建 user_info.behavior_type_mask
  - 改为在 meta_data.json 中保存 action_vocab
  - 新增 user_info.full_timestamp_seq
  - 供 dataloader 在训练时基于 full_action_seq 构造 task-specific token masks

依赖:
    pip install pandas numpy pyarrow
"""

import gc
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ================================================================
#  1. 常量 & 列定义
# ================================================================

LOG_FILES = [
    "log_random_4_22_to_5_08_27k.csv",
    "log_standard_4_08_to_4_21_27k_part1.csv",
    "log_standard_4_08_to_4_21_27k_part2.csv",
    "log_standard_4_22_to_5_08_27k_part1.csv",
    "log_standard_4_22_to_5_08_27k_part2.csv",
]

LOG_LOAD_COLUMNS = [
    "user_id", "video_id", "time_ms",
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view",
    "play_time_ms", "tab",
]

LOG_DTYPES = {
    "user_id": "int32",
    "video_id": "int32",
    "time_ms": "int64",
    "is_click": "int8",
    "is_like": "int8",
    "is_follow": "int8",
    "is_comment": "int8",
    "is_forward": "int8",
    "long_view": "int8",
    "play_time_ms": "int32",
    "tab": "int8",
}

LABEL_COLUMNS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view",
]

USER_STATIC_FEATURES = [
    "user_active_degree", "is_lowactive_period", "is_live_streamer",
    "is_video_author", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
] + [f"onehot_feat{i}" for i in range(18)]

CONTEXT_FEATURES = ["tab", "day_of_week", "is_weekend", "hour"]

ITEM_STATIC_FEATURES = ["video_type", "primary_tag", "music_type", "duration_bucket"]

FINAL_COLUMNS = (
    ["user_index", "item_index", "seq_len", "user_id"]
    + USER_STATIC_FEATURES + CONTEXT_FEATURES + LABEL_COLUMNS
)


# ================================================================
#  2. 工具函数
# ================================================================

def build_ordered_vocab(values, start=1):
    """从可迭代对象构建 vocab，start=1 时 0 预留给 padding/unknown。"""
    uniq = list(dict.fromkeys(str(v) for v in values))
    return {v: i for i, v in enumerate(uniq, start=start)}


PLAY_TIME_BINS = [-1, 0, 1000, 3000, 7000, 18000, 60000, float("inf")]
PLAY_TIME_LABELS = ["0", "0-1s", "1-3s", "3-7s", "7-18s", "18-60s", "60s+"]

DURATION_BINS = [-1, 0, 3000, 7000, 15000, 30000, 60000, float("inf")]
DURATION_LABELS = ["0", "0-3s", "3-7s", "7-15s", "15-30s", "30-60s", "60s+"]


def bucket_play_time(s: pd.Series) -> pd.Series:
    return pd.cut(s.fillna(0), bins=PLAY_TIME_BINS, labels=PLAY_TIME_LABELS).astype(str)


def bucket_duration(s: pd.Series) -> pd.Series:
    return pd.cut(s.fillna(0), bins=DURATION_BINS, labels=DURATION_LABELS).astype(str)


def fmt_date_int(d: int) -> str:
    return f"{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}"


def build_date_split(sorted_dates, train_ratio, valid_ratio, test_ratio):
    n_days = len(sorted_dates)
    if n_days < 3:
        raise ValueError(
            f"数据中仅有 {n_days} 个不同日期，至少需要 3 天才能按日期切分 train/valid/test。"
        )

    ratios = np.array([train_ratio, valid_ratio, test_ratio], dtype=np.float64)
    ratios = ratios / ratios.sum()

    raw = ratios * n_days
    counts = np.floor(raw).astype(int)
    rem = n_days - counts.sum()

    if rem > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for i in range(rem):
            counts[order[i % 3]] += 1

    for idx in range(3):
        while counts[idx] == 0:
            donors = np.where(counts > 1)[0]
            if len(donors) == 0:
                break
            donor = donors[np.argmax(counts[donors])]
            counts[donor] -= 1
            counts[idx] += 1

    n_train, n_valid, n_test = int(counts[0]), int(counts[1]), int(counts[2])

    train_dates = sorted_dates[:n_train]
    valid_dates = sorted_dates[n_train:n_train + n_valid]
    test_dates = sorted_dates[n_train + n_valid:]

    return {
        "n_days": n_days,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "train_dates": train_dates,
        "valid_dates": valid_dates,
        "test_dates": test_dates,
        "valid_start_date": valid_dates[0],
        "test_start_date": test_dates[0],
    }


# ================================================================
#  3. Vocab & 映射构建（不依赖日志扫描）
# ================================================================

def load_user_features(data_dir: Path) -> pd.DataFrame:
    """加载用户特征（27K 行，很小）。"""
    fp = data_dir / "user_features_27k.csv"
    print(f"  [Load] {fp.name}")
    cols = ["user_id"] + USER_STATIC_FEATURES
    uf = pd.read_csv(fp, usecols=cols)
    uf["user_id"] = uf["user_id"].astype("int32")
    for col in USER_STATIC_FEATURES:
        uf[col] = uf[col].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
    print(f"         {len(uf):,} users")
    return uf


def build_all_vocabs(uf: pd.DataFrame) -> dict:
    """从用户特征表 + 已知域构建所有类别特征 vocab（0=padding/unknown）。"""
    vocabs = {}
    for col in USER_STATIC_FEATURES:
        vals = sorted(set(uf[col].unique()) | {"__MISSING__"})
        vocabs[col] = {v: i + 1 for i, v in enumerate(vals)}
    vocabs["tab"] = {str(i): i + 1 for i in range(15)}
    vocabs["day_of_week"] = {str(i): i + 1 for i in range(7)}
    vocabs["is_weekend"] = {"0": 1, "1": 2}
    vocabs["hour"] = {str(i): i + 1 for i in range(24)}
    vocabs["play_time_bucket"] = {l: i + 1 for i, l in enumerate(PLAY_TIME_LABELS)}
    return vocabs


def build_action_maps():
    """枚举所有 2^6=64 种 action pattern → (pattern_int→name, name→code)。"""
    pat2name = {}
    for p in range(64):
        if p == 0:
            pat2name[p] = "exposure"
        else:
            parts = [c for i, c in enumerate(LABEL_COLUMNS) if p & (1 << i)]
            pat2name[p] = "|".join(parts)
    name2code = {n: i + 1 for i, n in enumerate(sorted(set(pat2name.values())))}
    return pat2name, name2code


def encode_user_features_to_int(uf: pd.DataFrame, vocabs: dict) -> pd.DataFrame:
    """将用户特征预编码为 int16，合并到日志时大幅节省内存。"""
    uf_enc = uf[["user_id"]].copy()
    for col in USER_STATIC_FEATURES:
        uf_enc[col] = uf[col].map(vocabs[col]).fillna(0).astype("int16")
    return uf_enc


# ================================================================
#  4. Phase 1: CSV → Partitioned Parquet
# ================================================================

def _preprocess_chunk(
    chunk: pd.DataFrame,
    uf_enc: pd.DataFrame,
    vocabs: dict,
) -> pd.DataFrame:
    """
    对单个 chunk (~2M 行) 执行:
      1. 时间戳分解 (保留 time_ms 供后续排序和 user_info.full_timestamp_seq)
      2. 提取 date 列 (int32, YYYYMMDD) 供按日期切分
      3. play_time 分桶 + 上下文特征编码 → int8
      4. 合并预编码用户特征 → int16
    """
    # ---- 时间戳分解 ----
    dt = pd.to_datetime(chunk["time_ms"], unit="ms", utc=True).dt.tz_convert(
        "Asia/Shanghai"
    )
    chunk["day_of_week"] = dt.dt.dayofweek.astype("int8")
    chunk["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    chunk["hour"] = dt.dt.hour.astype("int8")

    # ---- 提取日期列 (YYYYMMDD int32) ----
    chunk["date"] = (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).astype("int32")
    del dt

    # ---- play_time 分桶 + 编码 ----
    pt_bucket_str = bucket_play_time(chunk["play_time_ms"])
    chunk["play_time_bucket"] = (
        pt_bucket_str.map(vocabs["play_time_bucket"]).fillna(0).astype("int8")
    )
    del pt_bucket_str

    # ---- 上下文特征编码 ----
    chunk["tab"] = (
        chunk["tab"].astype(str).map(vocabs["tab"]).fillna(0).astype("int8")
    )
    chunk["day_of_week"] = (
        chunk["day_of_week"].astype(str).map(vocabs["day_of_week"]).fillna(0).astype("int8")
    )
    chunk["is_weekend"] = (
        chunk["is_weekend"].astype(str).map(vocabs["is_weekend"]).fillna(0).astype("int8")
    )
    chunk["hour"] = (
        chunk["hour"].astype(str).map(vocabs["hour"]).fillna(0).astype("int8")
    )

    # ---- 丢弃不再需要的列 ----
    chunk.drop(columns=["play_time_ms"], inplace=True)

    # ---- 合并预编码用户特征 (int16) ----
    chunk = chunk.merge(uf_enc, on="user_id", how="left")
    for col in USER_STATIC_FEATURES:
        chunk[col] = chunk[col].fillna(0).astype("int16")

    # ---- 去掉 ID 缺失行 ----
    chunk.dropna(subset=["user_id", "video_id"], inplace=True)

    return chunk


def phase1_partition_to_parquet(
    data_dir: Path,
    tmp_dir: Path,
    uf_enc: pd.DataFrame,
    vocabs: dict,
    n_parts: int,
    chunk_size: int,
    buffer_flush_size: int,
):
    """
    逐文件逐 chunk 读取 CSV → 预处理 → 按 user_id 哈希分区写入 Parquet。
    同时统计 user_counts / item_ids / all_dates。
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in range(n_parts):
        (tmp_dir / f"part_{p:03d}").mkdir(exist_ok=True)

    user_counts = defaultdict(int)
    item_ids = set()
    all_dates = set()

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
    for fname in LOG_FILES:
        print(f"  [Phase1] {fname}")
        reader = pd.read_csv(
            data_dir / fname,
            usecols=LOG_LOAD_COLUMNS,
            dtype=LOG_DTYPES,
            chunksize=chunk_size,
        )
        for chunk in reader:
            chunk = _preprocess_chunk(chunk, uf_enc, vocabs)

            vc = chunk["user_id"].value_counts(sort=False)
            for uid, cnt in zip(vc.index, vc.values):
                user_counts[uid] += int(cnt)
            item_ids.update(chunk["video_id"].unique().tolist())
            all_dates.update(chunk["date"].unique().tolist())

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
            del chunk, part_arr
            gc.collect()

        print(f"         累计 {total_rows:,} 行")

    for pid in range(n_parts):
        flush(pid)

    print(
        f"  [Phase1] 完成: {total_rows:,} 行, "
        f"{len(user_counts):,} 用户, {len(item_ids):,} 视频, "
        f"{len(all_dates)} 个不同日期\n"
    )
    return dict(user_counts), item_ids, all_dates


# ================================================================
#  5. Phase 2: Per-partition Process + Incremental Write
# ================================================================

def process_partition(
    tmp_dir: Path,
    pid: int,
    valid_users: set,
    user_idx_map: dict,
    item_idx_map: dict,
    pat2name: dict,
    name2code: dict,
    valid_start_date: int,
    test_start_date: int,
):
    """
    处理一个用户分区:
      1. 读取分区下所有 parquet 文件
      2. 过滤有效用户
      3. 按 (user_id, time_ms) 排序
      4. 编码 user_index / item_index / action / exposure
      5. 按日期切分 train/valid/test
      6. 构建 user_info 片段（保留 full_item_seq / full_action_seq / full_timestamp_seq）

    返回 dict{train, valid, test, user_info} 或 None。
    """
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

    # ---- 排序: 用户内按时间 ----
    df.sort_values(["user_id", "time_ms"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---- 编码 ID ----
    df["user_index"] = df["user_id"].map(user_idx_map).astype(np.int32)
    df["item_index"] = df["video_id"].map(item_idx_map).fillna(0).astype(np.int32)
    df["user_id"] = (df["user_index"] + 1).astype(np.int32)

    # ---- 标签 → float32 ----
    for col in LABEL_COLUMNS:
        df[col] = df[col].astype(np.float32)

    # ---- exposure + action ----
    binary = df[LABEL_COLUMNS].values.astype(np.int8)
    df["exposure"] = (binary.sum(axis=1) == 0).astype(np.float32)
    pattern = np.zeros(len(df), dtype=np.int32)
    for i in range(len(LABEL_COLUMNS)):
        pattern += binary[:, i].astype(np.int32) << i
    df["action"] = (
        pd.Series(pattern, index=df.index).map(pat2name).map(name2code).astype(np.int32)
    )
    del binary, pattern

    # ---- 特征转 int32 ----
    for col in USER_STATIC_FEATURES + CONTEXT_FEATURES:
        df[col] = df[col].astype(np.int32)

    # ---- time_ms 确保 int64 ----
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce").fillna(0).astype(np.int64)

    # ---- seq_len ----
    df["seq_len"] = df.groupby("user_index", sort=False).cumcount().astype(np.int32)

    # ---- 构建 user_info ----
    user_info_rows = []
    for uidx, gdf in df.groupby("user_index", sort=True):
        user_info_rows.append(
            {
                "user_index": int(uidx),
                "full_item_seq": gdf["item_index"].astype(int).tolist(),
                "full_action_seq": gdf["action"].astype(int).tolist(),
                "full_timestamp_seq": gdf["time_ms"].astype(np.int64).tolist(),
            }
        )

    # ---- 按日期区间切分 ----
    date_col = df["date"]
    train_df = df[date_col < valid_start_date]
    valid_df = df[(date_col >= valid_start_date) & (date_col < test_start_date)]
    test_df = df[date_col >= test_start_date]

    def _select_final(sdf):
        if len(sdf) == 0:
            return pd.DataFrame(columns=FINAL_COLUMNS)
        present = [c for c in FINAL_COLUMNS if c in sdf.columns]
        return sdf[present].reset_index(drop=True)

    result = {
        "train": _select_final(train_df),
        "valid": _select_final(valid_df),
        "test": _select_final(test_df),
        "user_info": user_info_rows,
    }

    del df, train_df, valid_df, test_df
    gc.collect()
    return result


# ================================================================
#  6. Item Info 构建
# ================================================================

def load_video_basic_features(data_dir: Path) -> pd.DataFrame:
    """加载视频基础特征，派生 primary_tag + duration_bucket。"""
    fp = data_dir / "video_features_basic_27k.csv"
    print(f"  [Load] {fp.name}")
    cols = ["video_id", "video_type", "music_type", "tag", "video_duration"]
    vf = pd.read_csv(fp, usecols=cols, dtype={"video_id": "int32"})
    vf["primary_tag"] = vf["tag"].astype(str).str.split(",").str[0].str.strip()
    vf["primary_tag"] = vf["primary_tag"].replace(
        {"nan": "__MISSING__", "": "__MISSING__"}
    )
    vf["duration_bucket"] = bucket_duration(vf["video_duration"].fillna(0))
    for col in ["video_type", "music_type"]:
        vf[col] = vf[col].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
    vf.drop(columns=["tag", "video_duration"], inplace=True)
    print(f"         {len(vf):,} videos")
    return vf


def build_item_info(data_dir: Path, item_idx_map: dict, output_dir: Path):
    """
    item_info.parquet:
      - 从 video_features_basic_27k.csv 加载
      - 仅保留日志中出现的视频
      - 编码 ITEM_STATIC_FEATURES
      - 第 0 行为 padding
    """
    vf = load_video_basic_features(data_dir)

    vf = vf[vf["video_id"].isin(item_idx_map)].copy()
    vf["item_index"] = vf["video_id"].map(item_idx_map).astype(np.int32)
    vf["item_id"] = vf["item_index"]
    vf.drop(columns=["video_id"], inplace=True)
    vf.drop_duplicates(subset=["item_index"], keep="last", inplace=True)

    item_vs = {}
    for col in ITEM_STATIC_FEATURES:
        vf[col] = vf[col].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
        vocab = build_ordered_vocab(vf[col], start=1)
        vf[col] = vf[col].map(vocab).astype(np.int32)
        item_vs[col] = len(vocab) + 1

    num_items = max(item_idx_map.values())
    item_info = pd.DataFrame(
        {
            "item_index": np.arange(num_items + 1, dtype=np.int32),
            "item_id": np.zeros(num_items + 1, dtype=np.int32),
            **{c: np.zeros(num_items + 1, dtype=np.int32) for c in ITEM_STATIC_FEATURES},
        }
    )

    vf_idx = vf.set_index("item_index")
    mask = item_info["item_index"].isin(vf_idx.index)
    matched = item_info.loc[mask, "item_index"].values
    for col in ["item_id"] + ITEM_STATIC_FEATURES:
        item_info.loc[mask, col] = vf_idx.loc[matched, col].values
    item_info = item_info.astype(np.int32)

    item_info.to_parquet(
        output_dir / "item_info.parquet", index=False, engine="pyarrow"
    )
    print(f"  [Saved] item_info.parquet ({num_items + 1:,} items incl. padding)")
    del vf, vf_idx
    return item_vs


# ================================================================
#  7. 主流程
# ================================================================

def preprocess_and_split(
    data_dir: str = "./KuaiRand-27K/data",
    output_dir: str = "./data/KuaiRand_27K_Action",
    min_user_interactions: int = 10,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_user_parts: int = 50,
    chunk_size: int = 2_000_000,
    buffer_flush_size: int = 500_000,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp_partitions"

    # ================================================================
    #  Step 1/8: 加载用户特征 + 构建 vocab
    # ================================================================
    print("\n[Step 1/8] 加载用户特征 & 构建 vocab")
    uf = load_user_features(data_dir)
    vocabs = build_all_vocabs(uf)
    pat2name, name2code = build_action_maps()
    uf_enc = encode_user_features_to_int(uf, vocabs)
    del uf
    print(f"  action 种类: {len(name2code)}")
    print("  用户特征已预编码为 int16\n")

    # ================================================================
    #  Step 2/8: Phase 1 — CSV → 分区 Parquet
    # ================================================================
    print("[Step 2/8] Phase 1: CSV → 分区 Parquet")
    user_counts, item_ids, all_dates = phase1_partition_to_parquet(
        data_dir,
        tmp_dir,
        uf_enc,
        vocabs,
        n_parts=n_user_parts,
        chunk_size=chunk_size,
        buffer_flush_size=buffer_flush_size,
    )
    del uf_enc
    gc.collect()

    # ================================================================
    #  Step 2.5/8: 按天数比例确定切分日期
    # ================================================================
    print("[Step 2.5/8] 按日期比例确定切分日期")
    sorted_dates = sorted(all_dates)
    split_info = build_date_split(
        sorted_dates,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
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
    print(f"  天数比例:  {split_info['n_train']}:{split_info['n_valid']}:{split_info['n_test']}\n")

    # ================================================================
    #  Step 3/8: 过滤低频用户 + 构建全局 ID 映射
    # ================================================================
    print("[Step 3/8] 过滤低频用户 + 构建 ID 映射")
    valid_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    n_dropped = len(user_counts) - len(valid_users)
    print(f"  有效用户: {len(valid_users):,}  (过滤 {n_dropped:,})")

    sorted_users = sorted(valid_users)
    user_idx_map = {u: i for i, u in enumerate(sorted_users)}

    sorted_items = sorted(item_ids)
    item_idx_map = {it: i + 1 for i, it in enumerate(sorted_items)}
    print(f"  视频数:   {len(item_idx_map):,}")

    vocab_size = {
        "user_index": len(user_idx_map),
        "item_index": len(item_idx_map) + 1,
        "user_id": len(user_idx_map) + 1,
        "item_id": len(item_idx_map) + 1,
        "action": len(name2code) + 1,
        "timestamp": 0,
    }
    for col in USER_STATIC_FEATURES + CONTEXT_FEATURES:
        vocab_size[col] = len(vocabs[col]) + 1

    del user_counts, item_ids, sorted_users, sorted_items
    gc.collect()
    print()

    # ================================================================
    #  Step 4/8: Phase 2 — 逐分区编码 + 按日期切分 + 增量写出
    # ================================================================
    print("[Step 4/8] 逐分区编码 + 按日期切分 + 增量写出")

    writers = {s: None for s in ["train", "valid", "test"]}
    all_user_info = []
    sample_counts = {"train": 0, "valid": 0, "test": 0}
    max_seq = 0

    for pid in range(n_user_parts):
        result = process_partition(
            tmp_dir,
            pid,
            valid_users,
            user_idx_map,
            item_idx_map,
            pat2name,
            name2code,
            valid_start_date,
            test_start_date,
        )
        if result is None:
            continue

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

        for row in result["user_info"]:
            sl = len(row["full_item_seq"])
            if sl > max_seq:
                max_seq = sl
        all_user_info.extend(result["user_info"])

        done = sum(sample_counts.values())
        print(f"  partition {pid + 1:3d}/{n_user_parts}: 累计 {done:,} 行")

        del result
        gc.collect()

    for w in writers.values():
        if w is not None:
            w.close()

    total = sum(sample_counts.values())
    print(
        f"\n  train={sample_counts['train']:,}  "
        f"valid={sample_counts['valid']:,}  "
        f"test={sample_counts['test']:,}"
    )
    print(f"  总计={total:,}\n")

    if total == 0:
        raise ValueError(
            "处理后数据为空，请检查原始数据或调小 min_user_interactions。"
        )

    del valid_users, user_idx_map
    gc.collect()

    # ================================================================
    #  Step 5/8: 保存 user_info
    # ================================================================
    print("[Step 5/8] 保存 user_info")
    num_users = vocab_size["user_index"]
    ui_dict = {r["user_index"]: r for r in all_user_info}

    user_info_df = pd.DataFrame(
        {
            "user_index": np.arange(num_users, dtype=np.int32),
            "full_item_seq": [
                ui_dict[i]["full_item_seq"] if i in ui_dict else []
                for i in range(num_users)
            ],
            "full_action_seq": [
                ui_dict[i]["full_action_seq"] if i in ui_dict else []
                for i in range(num_users)
            ],
            "full_timestamp_seq": [
                ui_dict[i]["full_timestamp_seq"] if i in ui_dict else []
                for i in range(num_users)
            ],
        }
    )
    user_info_df.to_parquet(
        output_dir / "user_info.parquet", index=False, engine="pyarrow"
    )
    print(f"  [Saved] user_info.parquet ({num_users:,} users)")
    del all_user_info, ui_dict, user_info_df
    gc.collect()

    # ================================================================
    #  Step 6/8: 构建 item_info
    # ================================================================
    print("\n[Step 6/8] 构建 item_info")
    item_vs = build_item_info(data_dir, item_idx_map, output_dir)
    del item_idx_map
    gc.collect()

    # ================================================================
    #  Step 7/8: 保存 meta_data
    # ================================================================
    print("\n[Step 7/8] 保存 meta_data")
    full_vs = dict(vocab_size)
    full_vs.update(item_vs)

    meta = {
        "sample_size": {
            "total": int(total),
            "train": int(sample_counts["train"]),
            "valid": int(sample_counts["valid"]),
            "test": int(sample_counts["test"]),
        },
        "split_by_date": {
            "train_days": split_info["n_train"],
            "train_range": f"{fmt_date_int(train_dates[0])} ~ {fmt_date_int(train_dates[-1])}",
            "valid_days": split_info["n_valid"],
            "valid_range": f"{fmt_date_int(valid_dates[0])} ~ {fmt_date_int(valid_dates[-1])}",
            "test_days": split_info["n_test"],
            "test_range": f"{fmt_date_int(test_dates[0])} ~ {fmt_date_int(test_dates[-1])}",
        },
        "vocab_size": {k: int(v) for k, v in full_vs.items()},
        "label": LABEL_COLUMNS,
        "action_vocab": {k: int(v) for k, v in name2code.items()},
        "action_vocab_desc": (
            "编码后的 action 词表，用于 dataloader 基于 full_action_seq "
            "构造 task-specific token masks。"
        ),
        "user_info_schema": {
            "fields": [
                "user_index",
                "full_item_seq",
                "full_action_seq",
                "full_timestamp_seq",
            ],
            "full_timestamp_seq_desc": "按时间顺序排列的原始 time_ms 序列",
        },
        "max_len": {
            "full_item_seq": int(max_seq),
            "full_action_seq": int(max_seq),
            "full_timestamp_seq": int(max_seq),
        },
    }

    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    print("  [Saved] meta_data.json")

    # ================================================================
    #  Step 8/8: 清理临时文件
    # ================================================================
    print("\n[Step 8/8] 清理临时分区文件")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  Done.")

    # ================================================================
    #  Summary
    # ================================================================
    print("\n" + "=" * 55)
    print("  KuaiRand-27K Preprocess Done (按日期比例切分)")
    print("=" * 55)
    print(f"输出目录: {output_dir}\n")
    for f in [
        "train.parquet",
        "valid.parquet",
        "test.parquet",
        "user_info.parquet",
        "item_info.parquet",
        "meta_data.json",
    ]:
        print(f"  ✔ {output_dir / f}")
    print("\nmeta_data.json:")
    print(json.dumps(meta, ensure_ascii=False, indent=4))


# ================================================================

if __name__ == "__main__":
    preprocess_and_split(
        data_dir="./KuaiRand-27K/data",
        output_dir="../KuaiRand_Video_Action",
        min_user_interactions=10,
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
        n_user_parts=20,
        chunk_size=2_000_000,
        buffer_flush_size=500_000,
    )