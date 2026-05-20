from datasets import load_dataset

cache_dir = "./hf_cache"

# Load item features
ds_item = load_dataset(
    "TAAC2025/TencentGR-10M",
    name="item_feat",
    split="train",
    cache_dir=cache_dir
)

# Load user behavior sequences
ds_seq = load_dataset(
    "TAAC2025/TencentGR-10M",
    name="seq",
    split="train",
    cache_dir=cache_dir
)

# Load user features
ds_user = load_dataset(
    "TAAC2025/TencentGR-10M",
    name="user_feat",
    split="train",
    cache_dir=cache_dir
)