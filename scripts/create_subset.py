"""从每个国家的标注数据中随机抽取 100 张图，生成 data2/ 用于快速测试。"""
import json, os, random, shutil
from pathlib import Path

random.seed(42)

PROJECT = Path(__file__).resolve().parents[1]
DATA1 = PROJECT / "data"
DATA2 = PROJECT / "data2"
IMAGES1 = DATA1 / "images"
IMAGES2 = DATA2 / "images"
IMAGES2.mkdir(parents=True, exist_ok=True)

# 每个国家的 labeled jsonl → 输出名
COUNTRIES = {
    "China_MotorBike_labeled.jsonl": "China_MotorBike_labeled.jsonl",
    "China_MotorBike_unlabeled.jsonl": "China_MotorBike_unlabeled.jsonl",
    "Czech_labeled.jsonl": "Czech_labeled.jsonl",
    "Czech_unlabeled.jsonl": "Czech_unlabeled.jsonl",
    "India_labeled.jsonl": "India_labeled.jsonl",
    "India_unlabeled.jsonl": "India_unlabeled.jsonl",
    "Japan_labeled.jsonl": "Japan_labeled.jsonl",
    "Japan_unlabeled.jsonl": "Japan_unlabeled.jsonl",
    "United_States_labeled.jsonl": "United_States_labeled.jsonl",
    "United_States_unlabeled.jsonl": "United_States_unlabeled.jsonl",
}

# 额外需要的最小 val/train 文件（从对应 labeled 复制）
SPLITS = {
    "china_motorbike_val.jsonl": "China_MotorBike_labeled.jsonl",
    "china_motorbike_train.jsonl": "China_MotorBike_labeled.jsonl",
    "czech_val.jsonl": "Czech_labeled.jsonl",
    "czech_train.jsonl": "Czech_labeled.jsonl",
    "japan_val.jsonl": "Japan_labeled.jsonl",
    "japan_train.jsonl": "Japan_labeled.jsonl",
}

SAMPLE_SIZE = 100
all_images = set()
stats = {}

print(f"从 {len(COUNTRIES)} 个 JSONL 各抽 {SAMPLE_SIZE} 条...")
for src_name, out_name in COUNTRIES.items():
    src = DATA1 / src_name
    if not src.exists():
        print(f"  SKIP {src_name} — 不存在")
        continue

    with open(src) as f:
        lines = f.readlines()

    if len(lines) <= SAMPLE_SIZE:
        sampled = lines
    else:
        sampled = random.sample(lines, SAMPLE_SIZE)

    # 写 JSONL
    out = DATA2 / out_name
    with open(out, "w") as f:
        for line in sampled:
            d = json.loads(line)
            all_images.add(d["image"])
            f.write(line)

    stats[out_name] = len(sampled)
    print(f"  OK {out_name}: {len(sampled)} 条")

# 生成 val/train 文件
print(f"\n生成 split 文件...")
for split_name, source_country in SPLITS.items():
    src = DATA2 / source_country
    if not src.exists():
        continue
    # 从对应国家中取 80% train, 20% val
    with open(src) as f:
        lines = [json.loads(l) for l in f.readlines()]

    n = len(lines)
    n_train = max(1, int(n * 0.8))
    n_val = n - n_train

    if "val" in split_name:
        out = DATA2 / split_name
        selected = lines[:n_val] if "val" in split_name else lines[n_val:]
        if "val" in split_name:
            selected = lines[:n_val]
            all_images.update(d["image"] for d in selected)
            with open(out, "w") as f:
                for d in selected:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            print(f"  OK {split_name}: {len(selected)} 条")
    elif "train" in split_name:
        out = DATA2 / split_name
        selected = lines[n_val:]
        all_images.update(d["image"] for d in selected)
        with open(out, "w") as f:
            for d in selected:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  OK {split_name}: {len(selected)} 条")

# 复制图片
print(f"\n复制 {len(all_images)} 张图片到 data2/images/ ...")
copied = 0
for img in sorted(all_images):
    src_img = IMAGES1 / img
    dst_img = IMAGES2 / img
    if src_img.exists() and not dst_img.exists():
        shutil.copy2(src_img, dst_img)
        copied += 1
    elif not src_img.exists():
        print(f"  MISSING: {img}")

print(f"\n完成: 复制了 {copied} 张图片")
print(f"JSONL 汇总:")
for name, count in sorted(stats.items()):
    print(f"  {name}: {count} 条")
