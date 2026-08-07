#!/usr/bin/env python3
"""Generate GAVE2 optic-disc masks, QC overlays, and a QC table.

This is the portable version of the script used for the preliminary validation
set.  It downloads the pinned public SegFormer checkpoint by default, combines
the optic-disc rim and cup classes, and retains one filled disc component.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


MODEL_ID = "pamixsun/segformer_for_optic_disc_cup_segmentation"
MODEL_REVISION = "e1698e9f52e24cb6a7b2fecab4688852b89f77ef"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing CFP images named g_NNN.png.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for binary 0/255 optic-disc masks.",
    )
    parser.add_argument(
        "--overlays",
        type=Path,
        required=True,
        help="Directory for green-contour visual QC overlays.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="Hugging Face model ID or a local model directory.",
    )
    parser.add_argument(
        "--revision",
        default=MODEL_REVISION,
        help="Pinned Hugging Face revision; ignored for a local model directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download model files. Useful for an offline review package.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=51,
        help="First expected GAVE2 image index.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=50,
        help="Required number of consecutive g_NNN.png images.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.001,
        help="QC lower bound for disc pixels / all image pixels.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.05,
        help="QC upper bound for disc pixels / all image pixels.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot see a GPU.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_expected_images(
    folder: Path, start_index: int, expected_count: int
) -> list[Path]:
    expected_names = {
        f"g_{index:03d}.png"
        for index in range(start_index, start_index + expected_count)
    }
    actual_names = {path.name for path in folder.glob("g_*.png")}
    if actual_names != expected_names:
        raise ValueError(
            f"{folder}: missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    return [folder / name for name in sorted(expected_names)]


def background_class_id(model: SegformerForSemanticSegmentation) -> int:
    labels = {
        int(class_id): str(label).lower()
        for class_id, label in model.config.id2label.items()
    }
    for class_id, label in labels.items():
        if "background" in label or label in {"bg", "back_ground"}:
            return class_id
    return 0


def keep_largest_filled_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if count <= 1:
        return np.zeros_like(binary, dtype=np.uint8)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = (labels == largest_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(largest)
    if contours:
        cv2.drawContours(
            filled, [max(contours, key=cv2.contourArea)], -1, 255, -1
        )
    return filled


def component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return count - 1


def make_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 5)
    return overlay


def model_source_args(args: argparse.Namespace) -> dict[str, object]:
    model_path = Path(args.model)
    options: dict[str, object] = {"local_files_only": args.local_files_only}
    if not model_path.is_dir():
        options["revision"] = args.revision
    return options


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_area_ratio < args.max_area_ratio <= 1.0:
        raise ValueError("Area-ratio bounds must satisfy 0 <= min < max <= 1.")

    image_paths = require_expected_images(
        args.images, args.start_index, args.expected_count
    )
    args.output.mkdir(parents=True, exist_ok=True)
    args.overlays.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    source_options = model_source_args(args)
    print(f"Loading {args.model} on {device}...")
    processor = AutoImageProcessor.from_pretrained(args.model, **source_options)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, **source_options
    ).to(device).eval()
    background_id = background_class_id(model)
    print("Class labels:", model.config.id2label)
    print("Background class:", background_id)

    qc_rows: list[dict[str, str]] = []
    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))

        inputs = {
            key: value.to(device)
            for key, value in processor(images=rgb, return_tensors="pt").items()
        }
        with torch.inference_mode():
            logits = model(**inputs).logits
            prediction = F.interpolate(
                logits,
                size=rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)[0].cpu().numpy()

        # The cup is inside the disc. Combining all non-background classes gives
        # the full optic-disc region required by the biomarker extractor.
        raw_disc = (prediction != background_id).astype(np.uint8) * 255
        disc_mask = keep_largest_filled_component(raw_disc)

        Image.fromarray(disc_mask, mode="L").save(args.output / image_path.name)
        Image.fromarray(make_overlay(rgb, disc_mask), mode="RGB").save(
            args.overlays / image_path.name
        )

        values = set(np.unique(disc_mask).tolist())
        components = component_count(disc_mask)
        area_ratio = float(np.count_nonzero(disc_mask) / disc_mask.size)
        valid = (
            disc_mask.shape == rgb.shape[:2]
            and values.issubset({0, 255})
            and components == 1
            and args.min_area_ratio <= area_ratio <= args.max_area_ratio
        )
        status = "ok" if valid else "review"
        qc_rows.append(
            {
                "file": image_path.name,
                "disc_area_ratio": f"{area_ratio:.6f}",
                "status": status,
            }
        )
        print(
            f"[{index:02d}/{len(image_paths)}] {image_path.name}: "
            f"{status}, components={components}, area={area_ratio:.6f}"
        )

    output_names = {path.name for path in args.output.glob("g_*.png")}
    input_names = {path.name for path in image_paths}
    if output_names != input_names:
        raise RuntimeError(
            f"Output-name mismatch: missing={sorted(input_names - output_names)}, "
            f"extra={sorted(output_names - input_names)}"
        )

    qc_path = args.output / "qc.csv"
    with qc_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("file", "disc_area_ratio", "status")
        )
        writer.writeheader()
        writer.writerows(qc_rows)

    review_count = sum(row["status"] == "review" for row in qc_rows)
    print(f"Masks: {args.output}")
    print(f"Overlays: {args.overlays}")
    print(f"QC table: {qc_path}")
    print(f"Checked: {len(qc_rows)}, review: {review_count}")


if __name__ == "__main__":
    main()
