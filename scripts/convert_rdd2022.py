"""Convert RDD2022 Pascal VOC XML annotations to PROBE JSONL manifest format.

Usage:
    python scripts/convert_rdd2022.py --xml-dir data/images --output-dir data

Produces:
    data/source_train.jsonl   — labeled training data (80% of labeled images)
    data/val.jsonl            — validation (20% of labeled images)
    data/target_unlabeled.jsonl — unlabeled target-domain images (test set)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# RDD2022 class name → 0-indexed label
CLASS_MAP = {
    "D00": 0,  # longitudinal crack
    "D10": 1,  # transverse crack
    "D20": 2,  # alligator crack
    "D40": 3,  # pothole
    "Repair": 4,
}


def parse_voc_xml(xml_path: Path) -> list[dict]:
    """Parse a Pascal VOC XML and return list of {"image", "boxes", "labels"}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    boxes = []
    labels = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASS_MAP:
            continue
        bbox = obj.find("bndbox")
        x1 = float(bbox.find("xmin").text)
        y1 = float(bbox.find("ymin").text)
        x2 = float(bbox.find("xmax").text)
        y2 = float(bbox.find("ymax").text)
        boxes.append([x1, y1, x2, y2])
        labels.append(CLASS_MAP[cls_name])
    return [{"image": filename, "boxes": boxes, "labels": labels}]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", default="data/images")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    xml_dir = Path(args.xml_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse all labeled samples
    xml_files = sorted(xml_dir.glob("*.xml"))
    all_labeled = []
    for xml_path in xml_files:
        all_labeled.extend(parse_voc_xml(xml_path))

    # Split into train / val
    indices = list(range(len(all_labeled)))
    random.shuffle(indices)
    val_count = int(len(indices) * args.val_frac)
    val_indices = set(indices[:val_count])
    train_indices = set(indices[val_count:])

    source_train = [all_labeled[i] for i in sorted(train_indices)]
    val = [all_labeled[i] for i in sorted(val_indices)]

    # Unlabeled target: all images without a corresponding XML
    all_images = set(p.stem for p in xml_dir.glob("*.jpg"))
    xml_images = set(p.stem for p in xml_files)
    unlabeled = sorted(all_images - xml_images)

    target_unlabeled = [{"image": f"{stem}.jpg", "boxes": [], "labels": []} for stem in unlabeled]

    write_jsonl(output_dir / "source_train.jsonl", source_train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "target_unlabeled.jsonl", target_unlabeled)

    print(f"source_train.jsonl:   {len(source_train)} samples")
    print(f"val.jsonl:            {len(val)} samples")
    print(f"target_unlabeled.jsonl: {len(target_unlabeled)} samples (no annotations)")
    print(f"Classes: {list(CLASS_MAP.keys())}")


if __name__ == "__main__":
    main()
