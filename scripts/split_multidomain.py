"""Split RDD2022 pre-packaged JSONL files into PROBE train/val/unlabeled manifests.

RDD2022 ships with per-country ``{Country}_labeled.jsonl`` and
``{Country}_unlabeled.jsonl`` files.  This script splits each labeled file
into train (80 %) / val (20 %) and copies the unlabeled file.

Usage
-----
  python scripts/split_multidomain.py --data-dir data
  python scripts/split_multidomain.py --data-dir data --val-frac 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split RDD2022 labeled JSONL into train/val, register domains"
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing *_labeled.jsonl files")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rng = random.Random(args.seed)

    # Discover labeled JSONL files
    labeled_files = sorted(data_dir.glob("*_labeled.jsonl"))
    if not labeled_files:
        print(f"No *_labeled.jsonl files found in {data_dir.resolve()}")
        print("Expected: data/Japan_labeled.jsonl, data/India_labeled.jsonl, ...")
        return

    registry = []

    for labeled_path in labeled_files:
        # Extract domain name: "Japan_labeled.jsonl" -> "japan"
        domain = labeled_path.stem.replace("_labeled", "").lower()
        unlabeled_path = data_dir / f"{labeled_path.stem.replace('_labeled', '_unlabeled')}.jsonl"

        # Read all labeled records
        with labeled_path.open("r", encoding="utf-8") as f:
            labeled_records = [json.loads(line) for line in f if line.strip()]

        # Shuffle and split
        indices = list(range(len(labeled_records)))
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * args.val_frac))
        val_indices = set(indices[:val_count])
        train_indices = set(indices[val_count:])

        train_records = [labeled_records[i] for i in sorted(train_indices)]
        val_records = [labeled_records[i] for i in sorted(val_indices)]

        # Write train/val
        train_path = data_dir / f"{domain}_train.jsonl"
        val_path = data_dir / f"{domain}_val.jsonl"
        write_jsonl(train_path, train_records)
        write_jsonl(val_path, val_records)

        # Unlabeled: copy or create
        unlabeled_out = data_dir / f"{domain}_unlabeled.jsonl"
        if unlabeled_path.exists():
            with unlabeled_path.open("r", encoding="utf-8") as f:
                unlabeled_records = [json.loads(line) for line in f if line.strip()]
            write_jsonl(unlabeled_out, unlabeled_records)
        else:
            # Images without annotations: scan for jpgs without entries in labeled
            image_root = data_dir / "images"
            if image_root.exists():
                all_jpgs = set(p.name for p in image_root.glob(f"{labeled_path.stem.replace('_labeled', '')}_*.jpg"))
                labeled_jpgs = {rec["image"] for rec in labeled_records}
                unlabeled_jpgs = sorted(all_jpgs - labeled_jpgs)
                unlabeled_records = [
                    {"image": name, "boxes": [], "labels": []} for name in unlabeled_jpgs
                ]
                write_jsonl(unlabeled_out, unlabeled_records)
            else:
                write_jsonl(unlabeled_out, [])

        n_train, n_val, n_unlabeled = len(train_records), len(val_records), len(unlabeled_records)
        print(
            f"  {domain:20s}: {n_train:5d} train, {n_val:4d} val, "
            f"{n_unlabeled:4d} unlabeled  ({n_train + n_val} labeled)"
        )

        registry.append({
            "name": domain,
            "train_manifest": f"{domain}_train.jsonl",
            "val_manifest": f"{domain}_val.jsonl",
            "unlabeled_manifest": f"{domain}_unlabeled.jsonl",
            "image_root": str((data_dir / "images").resolve()),
            "num_train": n_train,
            "num_val": n_val,
            "num_unlabeled": n_unlabeled,
            "num_labeled": n_train + n_val,
        })

    # Write domain registry
    registry_path = data_dir / "domains.json"
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print(f"\nDomain registry saved: {registry_path}")

    # Cross-domain pairs summary
    names = [d["name"] for d in registry]
    pairs = [(s, t) for s in names for t in names if s != t]
    print(f"Cross-domain pairs: {len(pairs)} total")
    for src, tgt in pairs:
        print(f"  {src} -> {tgt}")


if __name__ == "__main__":
    main()
