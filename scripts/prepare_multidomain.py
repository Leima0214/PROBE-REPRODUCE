"""Prepare multi-domain RDD2022 data for PROBE cross-domain experiments.

Scans the data directory, auto-detects domains from file naming or directory
structure, and generates JSONL manifests for each domain plus a domain registry.

Supported layouts
-----------------
  Flat:    data/images/{Domain}_*.jpg + data/images/{Domain}_*.xml
  Nested:  data/{Domain}/images/*.jpg + data/{Domain}/annotations/*.xml

Output
------
  data/{domain}_train.jsonl       — 80% of labeled images
  data/{domain}_val.jsonl         — 20% of labeled images
  data/{domain}_unlabeled.jsonl   — images without XML annotations
  data/domains.json               — registry of all detected domains

Usage
-----
  python scripts/prepare_multidomain.py --data-root data/images
  python scripts/prepare_multidomain.py --data-root data --layout nested
  python scripts/prepare_multidomain.py --data-root data/images --val-frac 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# RDD2022 class name -> 0-indexed label
CLASS_MAP = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
    "Repair": 4,
}

# Known RDD2022 domain name patterns — used for extraction from filenames
# Format: (regex_pattern, canonical_name)
DOMAIN_PATTERNS = [
    (r"China_MotorBike", "china_motorbike"),
    (r"China_Drone", "china_drone"),
    (r"Japan", "japan"),
    (r"India", "india"),
    (r"Czech", "czech"),
    (r"United_States", "united_states"),
    (r"Norway", "norway"),
]


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

def extract_domain_flat(filename_stem: str) -> str | None:
    """Extract domain from a flat-layout filename like 'China_MotorBike_000001'."""
    for pattern, canonical in DOMAIN_PATTERNS:
        if re.match(pattern, filename_stem, re.IGNORECASE):
            return canonical
    return None


def extract_domain_nested(xml_path: Path, data_root: Path) -> str | None:
    """Extract domain from nested layout: data/{Domain}/annotations/*.xml.

    Only matches when the XML is inside an 'annotations' (or 'xmls') subdirectory.
    """
    parts = xml_path.relative_to(data_root).parts
    # Expected nested: {domain}/annotations/{file}.xml  or  {domain}/xmls/{file}.xml
    if len(parts) >= 2 and parts[-2].lower() in ("annotations", "xmls", "labels"):
        return parts[0].lower().replace(" ", "_")
    return None


def detect_layout_and_domains(data_root: Path) -> tuple[str, dict[str, dict]]:
    """Auto-detect layout and group XML files by domain.

    Returns
    -------
    layout : str — "flat" or "nested"
    domains : dict[canonical_name, {"xml_files": [Path, ...], "image_root": Path}]
    """
    xml_files = sorted(data_root.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under {data_root}")

    # Try nested layout first: XMLs inside an annotations/xmls/labels subdirectory
    nested_domains = defaultdict(list)
    for xf in xml_files:
        domain = extract_domain_nested(xf, data_root)
        if domain:
            nested_domains[domain].append(xf)

    # If nested detection found domains, use nested layout
    if len(nested_domains) > 0:
        domains = {}
        for domain, files in nested_domains.items():
            domain_dir = files[0].parents[1]  # {domain}/annotations -> {domain}
            img_dir = domain_dir / "images"
            if not img_dir.exists():
                img_dir = domain_dir
            domains[domain] = {"xml_files": sorted(files), "image_root": img_dir}
        print(f"Detected nested layout: {len(domains)} domains")
        return "nested", domains

    # Fall back to flat layout: extract domain from filename prefix
    flat_domains = defaultdict(list)
    for xf in xml_files:
        domain = extract_domain_flat(xf.stem)
        if domain:
            flat_domains[domain].append(xf)
        else:
            print(f"  [warn] Could not detect domain for: {xf.name}")

    if not flat_domains:
        raise RuntimeError(
            "Could not detect any known RDD2022 domains. "
            f"Expected filename patterns: {[p for p, _ in DOMAIN_PATTERNS]}"
        )

    domains = {}
    for domain, files in flat_domains.items():
        domains[domain] = {"xml_files": sorted(files), "image_root": data_root}
    print(f"Detected flat layout: {len(domains)} domains")
    return "flat", domains


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_voc_xml(xml_path: Path) -> list[dict]:
    """Parse a Pascal VOC XML and return records with image/boxes/labels."""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-domain split
# ---------------------------------------------------------------------------

def prepare_domain(
    domain: str,
    info: dict,
    output_dir: Path,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """Generate train/val/unlabeled JSONL manifests for one domain.

    Returns domain metadata for the registry.
    """
    rng = random.Random(seed)
    xml_files = info["xml_files"]
    image_root = info["image_root"]

    # Parse all labeled samples
    all_labeled = []
    for xf in xml_files:
        all_labeled.extend(parse_voc_xml(xf))

    # Shuffle and split
    indices = list(range(len(all_labeled)))
    rng.shuffle(indices)
    val_count = max(1, int(len(indices) * val_frac))
    val_indices = set(indices[:val_count])
    train_indices = set(indices[val_count:])

    train_records = [all_labeled[i] for i in sorted(train_indices)]
    val_records = [all_labeled[i] for i in sorted(val_indices)]

    # Unlabeled: images without XML
    all_jpgs = set(p.stem for p in image_root.glob("*.jpg"))
    xml_stems = set(xf.stem for xf in xml_files)
    unlabeled_stems = sorted(all_jpgs - xml_stems)
    unlabeled_records = [
        {"image": f"{stem}.jpg", "boxes": [], "labels": []}
        for stem in unlabeled_stems
    ]

    # Write JSONL
    write_jsonl(output_dir / f"{domain}_train.jsonl", train_records)
    write_jsonl(output_dir / f"{domain}_val.jsonl", val_records)
    write_jsonl(output_dir / f"{domain}_unlabeled.jsonl", unlabeled_records)

    n_labeled = len(train_records) + len(val_records)
    print(
        f"  {domain:20s}: {len(train_records):4d} train, "
        f"{len(val_records):4d} val, "
        f"{len(unlabeled_records):4d} unlabeled  "
        f"({n_labeled} labeled images, {len(xml_files)} xmls)"
    )

    return {
        "name": domain,
        "train_manifest": f"{domain}_train.jsonl",
        "val_manifest": f"{domain}_val.jsonl",
        "unlabeled_manifest": f"{domain}_unlabeled.jsonl",
        "image_root": str(image_root.resolve()),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "num_unlabeled": len(unlabeled_records),
        "num_labeled": n_labeled,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare multi-domain RDD2022 data for PROBE experiments"
    )
    parser.add_argument(
        "--data-root", default="data/images",
        help="Root directory containing images/XMLs (flat) or domain subdirs (nested)",
    )
    parser.add_argument("--output-dir", default="data", help="Where to write JSONL manifests")
    parser.add_argument("--layout", choices=["auto", "flat", "nested"], default="auto")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect layout and domains
    print(f"Scanning: {data_root.resolve()}")
    layout, domains = detect_layout_and_domains(data_root)
    if args.layout != "auto":
        layout = args.layout
    print(f"Layout: {layout}")
    print(f"Domains found: {list(domains.keys())}")
    print()

    # Prepare each domain
    registry = []
    for domain_name in sorted(domains.keys()):
        meta = prepare_domain(
            domain_name,
            domains[domain_name],
            output_dir,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        registry.append(meta)

    # Write domain registry
    registry_path = output_dir / "domains.json"
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print(f"\nDomain registry saved: {registry_path}")

    # Print cross-domain experiment pairs
    names = [d["name"] for d in registry]
    pairs = [(s, t) for s in names for t in names if s != t]
    print(f"\nCross-domain pairs ({len(pairs)} total):")
    for src, tgt in pairs:
        print(f"  {src} -> {tgt}")


if __name__ == "__main__":
    main()
