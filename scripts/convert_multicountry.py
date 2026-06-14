"""Convert multi-country RDD2022 Pascal VOC XML annotations to per-country JSONL manifests.

Usage:
    python scripts/convert_multicountry.py

Produces per-country files in data/:
    data/<country>_labeled.jsonl   — all labeled images (80/20 train/val split)
    data/<country>_unlabeled.jsonl — all images without annotations (test set)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# RDD2022 class name → 0-indexed label
CLASS_MAP = {
    "D00": 0,   # longitudinal crack
    "D10": 1,   # transverse crack
    "D20": 2,   # alligator crack
    "D40": 3,   # pothole
    "Repair": 4,
}

# Country prefixes from RDD2022 filenames
COUNTRIES = [
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "United_States",
]


def parse_voc_xml(xml_path: Path) -> dict:
    """Parse a Pascal VOC XML and return {"image", "boxes", "labels"}."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    boxes, labels = [], []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASS_MAP:
            continue
        bbox = obj.find("bndbox")
        boxes.append([
            float(bbox.find("xmin").text),
            float(bbox.find("ymin").text),
            float(bbox.find("xmax").text),
            float(bbox.find("ymax").text),
        ])
        labels.append(CLASS_MAP[cls_name])
    return {"image": filename, "boxes": boxes, "labels": labels}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files per country
    country_images: dict[str, set[str]] = defaultdict(set)
    country_xmls: dict[str, set[str]] = defaultdict(set)

    for jpg_path in sorted(image_dir.glob("*.jpg")):
        stem = jpg_path.stem
        for country in COUNTRIES:
            if stem.startswith(country + "_"):
                country_images[country].add(stem)
                break

    for xml_path in sorted(image_dir.glob("*.xml")):
        stem = xml_path.stem
        for country in COUNTRIES:
            if stem.startswith(country + "_"):
                country_xmls[country].add(stem)
                break

    for country in COUNTRIES:
        if not country_images[country]:
            print(f"  SKIP {country} — no images found")
            continue

        # Parse labeled samples (have XML)
        labeled = []
        for stem in sorted(country_xmls[country]):
            xml_path = image_dir / f"{stem}.xml"
            labeled.append(parse_voc_xml(xml_path))

        # Unlabeled = all images minus those with XML
        unlabeled_stems = sorted(country_images[country] - country_xmls[country])
        unlabeled = [
            {"image": f"{stem}.jpg", "boxes": [], "labels": []}
            for stem in unlabeled_stems
        ]

        # Write per-country files
        write_jsonl(output_dir / f"{country}_labeled.jsonl", labeled)
        write_jsonl(output_dir / f"{country}_unlabeled.jsonl", unlabeled)

        print(
            f"  {country}: {len(labeled)} labeled, "
            f"{len(unlabeled)} unlabeled "
            f"(total {len(country_images[country])} images)"
        )

    # Also generate source_train / val / target_unlabeled for China_MotorBike
    # (matches existing convention for backward compatibility)
    print("\n=== Generating default manifests (China_MotorBike as source) ===")
    import random
    random.seed(args.seed)
    china_labeled = []
    for stem in sorted(country_xmls.get("China_MotorBike", set())):
        china_labeled.append(parse_voc_xml(image_dir / f"{stem}.xml"))
    indices = list(range(len(china_labeled)))
    random.shuffle(indices)
    val_count = int(len(indices) * 0.2)
    val_indices = set(indices[:val_count])
    train_indices = set(indices[val_count:])

    source_train = [china_labeled[i] for i in sorted(train_indices)]
    val = [china_labeled[i] for i in sorted(val_indices)]
    china_unlabeled_stems = sorted(
        country_images.get("China_MotorBike", set()) - country_xmls.get("China_MotorBike", set())
    )
    target_unlabeled = [
        {"image": f"{stem}.jpg", "boxes": [], "labels": []}
        for stem in china_unlabeled_stems
    ]

    write_jsonl(output_dir / "source_train.jsonl", source_train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "target_unlabeled.jsonl", target_unlabeled)
    print(f"  source_train.jsonl:       {len(source_train)} samples")
    print(f"  val.jsonl:                {len(val)} samples")
    print(f"  target_unlabeled.jsonl:   {len(target_unlabeled)} samples")


if __name__ == "__main__":
    main()
