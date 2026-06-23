"""检查上传到 V100 的数据是否完整——JSONL 记录 vs 实际图片文件。"""
import json, sys, os
from pathlib import Path

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/data")

REQUIRED_JSONL = [
    "China_MotorBike_labeled.jsonl",
    "Japan_unlabeled.jsonl",
    "japan_val.jsonl",
]
OPTIONAL_JSONL = [
    "China_MotorBike_unlabeled.jsonl",
    "china_motorbike_val.jsonl",
    "china_motorbike_train.jsonl",
]

errors = []

# 1. 检查 JSONL 文件
print("=" * 60)
print("1. JSONL 文件检查")
print("=" * 60)
for fname in REQUIRED_JSONL:
    path = DATA_DIR / fname
    if not path.exists():
        errors.append(f"缺少必需文件: {fname}")
        print(f"  ❌ {fname} — 缺失")
    else:
        size_mb = path.stat().st_size / 1024
        with open(path) as f:
            lines = f.readlines()
        print(f"  ✅ {fname} — {len(lines)} 条记录, {size_mb:.0f}KB")

for fname in OPTIONAL_JSONL:
    path = DATA_DIR / fname
    if path.exists():
        with open(path) as f:
            lines = f.readlines()
        print(f"  ✅ {fname} (可选) — {len(lines)} 条记录")

# 2. 检查 JSONL 引用的图片文件
print("\n" + "=" * 60)
print("2. JSONL 引用的图片检查")
print("=" * 60)

all_images_needed = set()
jsonl_files = REQUIRED_JSONL.copy()
for fname in OPTIONAL_JSONL:
    if (DATA_DIR / fname).exists():
        jsonl_files.append(fname)

total_records = 0
for fname in jsonl_files:
    path = DATA_DIR / fname
    if not path.exists():
        continue
    with open(path) as f:
        for line in f:
            total_records += 1
            try:
                d = json.loads(line)
                img = d.get("image", "")
                if img:
                    all_images_needed.add(img)
            except json.JSONDecodeError:
                pass

print(f"  总 JSONL 记录: {total_records}")
print(f"  引用的唯一图片: {len(all_images_needed)}")

images_dir = DATA_DIR / "images"
if not images_dir.exists():
    errors.append("images 目录不存在！")
    print(f"  ❌ images/ 目录缺失")
else:
    existing = set(os.listdir(images_dir))
    missing_imgs = all_images_needed - existing
    extra_imgs = [f for f in existing if f not in all_images_needed and (f.endswith(".jpg") or f.endswith(".png"))]

    if missing_imgs:
        pct = 100 * len(missing_imgs) / len(all_images_needed)
        errors.append(f"缺少 {len(missing_imgs)} 张图片 ({pct:.1f}%)")
        print(f"  ❌ 缺少 {len(missing_imgs)} 张图片 ({pct:.1f}%):")
        for img in sorted(missing_imgs)[:10]:
            print(f"       {img}")
        if len(missing_imgs) > 10:
            print(f"       ... 还有 {len(missing_imgs) - 10} 张")
    else:
        print(f"  ✅ 所有 {len(all_images_needed)} 张图片存在")

    if extra_imgs:
        print(f"  ℹ️  多余 {len(extra_imgs)} 张图片（未被 JSONL 引用）")

    print(f"  磁盘使用: images/ 共 {len(existing)} 个文件")

# 3. 数据集统计
print("\n" + "=" * 60)
print("3. 数据集统计")
print("=" * 60)

for fname in jsonl_files:
    path = DATA_DIR / fname
    if not path.exists():
        continue
    with_boxes = 0
    total = 0
    classes = {}
    with open(path) as f:
        for line in f:
            total += 1
            d = json.loads(line)
            boxes = d.get("boxes", [])
            if boxes:
                with_boxes += 1
                for box in boxes:
                    cls_id = box.get("label", box.get("class", -1))
                    classes[cls_id] = classes.get(cls_id, 0) + 1

    label = "有标注" if with_boxes > 0 else "无标注(SSL用)"
    print(f"\n  {fname}:")
    print(f"    总数: {total}, {label}: {with_boxes}")
    if classes:
        print(f"    各类别框数: {dict(sorted(classes.items()))}")

# 汇总
print("\n" + "=" * 60)
if errors:
    print(f"❌ 发现 {len(errors)} 个问题:")
    for e in errors:
        print(f"   - {e}")
else:
    print("✅ 数据完整性检查通过，所有文件就绪")
print("=" * 60)
