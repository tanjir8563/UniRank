from pathlib import Path
from datasets import load_dataset

def export_one(config_name: str, cache_dir: Path, out_dir: Path):
    print(f"\nLoading split: {config_name}")
    ds = load_dataset(
        "TAAC2025/TencentGR-10M",
        name=config_name,
        split="train",
        cache_dir=str(cache_dir)
    )

    print(f"{config_name}: {len(ds)} rows")

    output_file = out_dir / f"{config_name}.parquet"
    print(f"Exporting to: {output_file}")
    ds.to_parquet(str(output_file))
    print(f"Finished: {output_file}")

def main():
    base_dir = Path(__file__).resolve().parent
    cache_dir = base_dir / "hf_cache"
    out_dir = base_dir

    for config_name in ["item_feat", "seq", "user_feat"]:
        try:
            export_one(config_name, cache_dir, out_dir)
        except Exception as e:
            print(f"Failed to export {config_name}: {e}")

    print("\nAll done.")

if __name__ == "__main__":
    main()