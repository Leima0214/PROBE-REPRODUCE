from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class RoadDamageDataset(Dataset):
    """Road-damage detection dataset backed by a JSONL manifest.

    Each labeled line should contain:
    ``{"image": "x.jpg", "boxes": [[x1,y1,x2,y2]], "labels": [1]}``
    Unlabeled target-domain samples may omit ``boxes`` and ``labels``.
    """

    def __init__(
        self,
        manifest: str | Path,
        image_root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        with Path(manifest).open("r", encoding="utf-8") as handle:
            self.samples = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(self.image_root / sample["image"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = {
            "boxes": torch.tensor(sample.get("boxes", []), dtype=torch.float32),
            "labels": torch.tensor(sample.get("labels", []), dtype=torch.long),
            "image_id": torch.tensor([index]),
        }
        return image, target
