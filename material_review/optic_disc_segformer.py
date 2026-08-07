#!/usr/bin/env python3
"""Generate binary optic-disc masks with the pinned public SegFormer model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    processor = AutoImageProcessor.from_pretrained(
        args.model, local_files_only=True
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, local_files_only=True
    ).eval().to(device)

    args.output.mkdir(parents=True, exist_ok=True)
    images = sorted(args.images.glob("g_*.png"))
    if args.limit is not None:
        images = images[: args.limit]
    for index, path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] optic disc: {path.name}", flush=True)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = {
            key: value.to(device)
            for key, value in processor(image, return_tensors="pt").items()
        }
        with torch.inference_mode():
            logits = model(**inputs).logits
            labels = F.interpolate(
                logits,
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)[0]
        # Both label 1 (disc rim) and label 2 (cup) belong to the optic disc.
        mask = (labels > 0).to(torch.uint8).cpu().numpy() * 255
        Image.fromarray(mask, mode="L").save(args.output / path.name)


if __name__ == "__main__":
    main()
