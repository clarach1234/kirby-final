#!/usr/bin/env python3
"""Build the 2026-07-30 three-task exploratory GAVE2 submission.

The generated candidate uses:

* Task 1: equal five-fold ensemble with a logit shift that maps the original
  probability threshold 0.30 to the evaluator boundary 0.50.
* Task 2: the D ensemble with additive-only vein skeleton recovery. Existing
  probabilities are never reduced or zeroed.
* Task 3: the E result with conservative, distribution-informed calibration.

The submission ZIP contains only Task1/, Task2/, and Task3/. A detailed JSON
manifest is written next to the ZIP so it cannot interfere with evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import binary_dilation, disk, skeletonize


EXPECTED_STEMS = tuple(f"g_{index:03d}" for index in range(51, 101))
TASK3_KEYS = (
    "CRAE",
    "CRVE",
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)

TASK3_TRAIN_BOUNDS = {
    "AVR": (0.437141, 0.910111),
    "artery_density": (0.017965, 0.047736),
    "artery_fractal_dimension": (1.268001, 1.491455),
    "vein_fractal_dimension": (1.294064, 1.470247),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task1-source",
        type=Path,
        default=Path("task1_01234_submission"),
    )
    parser.add_argument(
        "--task2-source",
        type=Path,
        default=Path("task2_D_submission"),
    )
    parser.add_argument(
        "--task3-source",
        type=Path,
        default=Path("task3_E_submission"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "submission_experiments/"
            "kirby_exp_t1eq030_t2D_veinskel_t3E_cal"
        ),
    )
    parser.add_argument("--task1-threshold", type=float, default=0.30)
    parser.add_argument("--task2-vein-core", type=float, default=0.50)
    parser.add_argument("--task2-vein-support", type=float, default=0.20)
    parser.add_argument("--task2-vein-dilation-support", type=float, default=0.10)
    parser.add_argument("--task2-added-confidence", type=float, default=0.60)
    parser.add_argument("--task3-avr-offset", type=float, default=-0.010)
    parser.add_argument("--task3-artery-density-center", type=float, default=0.03174888)
    parser.add_argument("--task3-artery-density-scale", type=float, default=1.30)
    parser.add_argument("--task3-artery-fd-offset", type=float, default=-0.013)
    parser.add_argument("--task3-vein-fd-offset", type=float, default=0.012)
    return parser.parse_args()


def require_probability(value: float, label: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{label} must be strictly between 0 and 1: {value}")


def validate_sources(args: argparse.Namespace) -> None:
    for value, label in (
        (args.task1_threshold, "task1_threshold"),
        (args.task2_vein_core, "task2_vein_core"),
        (args.task2_vein_support, "task2_vein_support"),
        (args.task2_vein_dilation_support, "task2_vein_dilation_support"),
        (args.task2_added_confidence, "task2_added_confidence"),
    ):
        require_probability(value, label)

    if not (
        args.task2_vein_dilation_support
        <= args.task2_vein_support
        < args.task2_vein_core
    ):
        raise ValueError(
            "Task 2 thresholds must satisfy dilation_support <= support < core"
        )
    if args.task2_added_confidence <= 0.5:
        raise ValueError("Added Task 2 skeleton confidence must exceed 0.5")

    expected_pngs = {f"{stem}.png" for stem in EXPECTED_STEMS}
    expected_txts = {f"{stem}.txt" for stem in EXPECTED_STEMS}
    for source, expected, suffix in (
        (args.task1_source, expected_pngs, ".png"),
        (args.task2_source, expected_pngs, ".png"),
        (args.task3_source, expected_txts, ".txt"),
    ):
        if not source.is_dir():
            raise FileNotFoundError(f"Missing source directory: {source}")
        actual = {path.name for path in source.glob(f"*{suffix}")}
        if actual != expected:
            raise ValueError(
                f"{source}: expected exactly {len(expected)} files; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {args.output}"
        )
    zip_path = args.output.with_suffix(".zip")
    manifest_path = args.output.with_name(f"{args.output.name}_manifest.json")
    if zip_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {zip_path} or {manifest_path}"
        )


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (1536, 1024):
            raise ValueError(f"{path}: expected 1536x1024, got {image.size}")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb(array: np.ndarray, path: Path) -> None:
    if array.shape != (1024, 1536, 3) or array.dtype != np.uint8:
        raise ValueError(
            f"Invalid output array for {path}: shape={array.shape}, dtype={array.dtype}"
        )
    Image.fromarray(array, mode="RGB").save(path)


def shift_probability_threshold(array: np.ndarray, threshold: float) -> np.ndarray:
    """Map the original probability ``threshold`` to 0.5 via a logit shift."""
    probability = array.astype(np.float32) / 255.0
    numerator = probability * (1.0 - threshold)
    denominator = numerator + (1.0 - probability) * threshold
    shifted = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return np.rint(np.clip(shifted, 0.0, 1.0) * 255.0).astype(np.uint8)


def process_task1(args: argparse.Namespace, destination: Path) -> dict[str, object]:
    channel_names = ("artery", "vessel", "vein")
    before_positive = np.zeros(3, dtype=np.int64)
    after_positive = np.zeros(3, dtype=np.int64)

    destination.mkdir(parents=True)
    for stem in EXPECTED_STEMS:
        source = load_rgb(args.task1_source / f"{stem}.png")
        shifted = shift_probability_threshold(source, args.task1_threshold)
        before_positive += np.sum(source >= 128, axis=(0, 1))
        after_positive += np.sum(shifted >= 128, axis=(0, 1))
        save_rgb(shifted, destination / f"{stem}.png")

    return {
        "source": str(args.task1_source.resolve()),
        "operation": "logit_threshold_shift",
        "original_threshold_mapped_to_0_5": args.task1_threshold,
        "positive_pixels_before": dict(
            zip(channel_names, (int(value) for value in before_positive))
        ),
        "positive_pixels_after": dict(
            zip(channel_names, (int(value) for value in after_positive))
        ),
        "positive_pixel_ratio_after_over_before": {
            name: float(after / before)
            for name, before, after in zip(
                channel_names, before_positive, after_positive
            )
        },
    }


def recover_vein_skeleton(
    image: np.ndarray,
    core_threshold: float,
    support_threshold: float,
    dilation_support_threshold: float,
    added_confidence: float,
) -> tuple[np.ndarray, dict[str, int]]:
    probability = image.astype(np.float32) / 255.0
    vein = probability[:, :, 2]

    core = vein >= core_threshold
    support = vein >= support_threshold
    support_skeleton = skeletonize(support)

    labels, component_count = ndi.label(
        support_skeleton,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    touched_labels = np.unique(labels[np.logical_and(support_skeleton, core)])
    touched_labels = touched_labels[touched_labels > 0]
    selected_skeleton = np.isin(labels, touched_labels)

    dilated = binary_dilation(selected_skeleton, footprint=disk(1))
    added = (
        dilated
        & ~core
        & (vein >= dilation_support_threshold)
    )

    result = probability.copy()
    result[:, :, 2][added] = np.maximum(
        result[:, :, 2][added],
        added_confidence,
    )
    # Preserve semantic consistency locally without applying the global Db boost.
    result[:, :, 1][added] = np.maximum(
        result[:, :, 1][added],
        result[:, :, 2][added],
    )

    output = np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)
    return output, {
        "support_components": int(component_count),
        "core_connected_components": int(len(touched_labels)),
        "support_skeleton_pixels": int(support_skeleton.sum()),
        "selected_skeleton_pixels": int(selected_skeleton.sum()),
        "added_pixels": int(added.sum()),
        "core_pixels": int(core.sum()),
    }


def process_task2(args: argparse.Namespace, destination: Path) -> dict[str, object]:
    totals = {
        "support_components": 0,
        "core_connected_components": 0,
        "support_skeleton_pixels": 0,
        "selected_skeleton_pixels": 0,
        "added_pixels": 0,
        "core_pixels": 0,
    }
    per_image_added: dict[str, int] = {}

    destination.mkdir(parents=True)
    for stem in EXPECTED_STEMS:
        source = load_rgb(args.task2_source / f"{stem}.png")
        recovered, stats = recover_vein_skeleton(
            source,
            core_threshold=args.task2_vein_core,
            support_threshold=args.task2_vein_support,
            dilation_support_threshold=args.task2_vein_dilation_support,
            added_confidence=args.task2_added_confidence,
        )
        save_rgb(recovered, destination / f"{stem}.png")
        for key, value in stats.items():
            totals[key] += value
        per_image_added[stem] = stats["added_pixels"]

    return {
        "source": str(args.task2_source.resolve()),
        "operation": "additive_core_connected_vein_skeleton_recovery",
        "core_threshold": args.task2_vein_core,
        "support_threshold": args.task2_vein_support,
        "dilation_support_threshold": args.task2_vein_dilation_support,
        "added_confidence": args.task2_added_confidence,
        "dilation_radius_pixels": 1,
        "existing_probabilities_reduced": False,
        "totals": totals,
        "added_over_core_ratio": (
            float(totals["added_pixels"] / totals["core_pixels"])
            if totals["core_pixels"]
            else 0.0
        ),
        "per_image_added_pixels": per_image_added,
    }


def read_biomarkers(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}: malformed line: {line!r}")
        key, raw_value = parts
        if key in values:
            raise ValueError(f"{path}: duplicate biomarker: {key}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite biomarker: {line!r}")
        values[key] = value
    if set(values) != set(TASK3_KEYS):
        raise ValueError(
            f"{path}: biomarker keys differ; "
            f"missing={set(TASK3_KEYS) - set(values)}, "
            f"extra={set(values) - set(TASK3_KEYS)}"
        )
    return values


def clipped(value: float, key: str) -> float:
    lower, upper = TASK3_TRAIN_BOUNDS[key]
    return float(np.clip(value, lower, upper))


def calibrate_biomarkers(
    values: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    calibrated = dict(values)
    calibrated["AVR"] = clipped(
        values["AVR"] + args.task3_avr_offset,
        "AVR",
    )
    calibrated["artery_density"] = clipped(
        args.task3_artery_density_center
        + args.task3_artery_density_scale
        * (values["artery_density"] - args.task3_artery_density_center),
        "artery_density",
    )
    calibrated["artery_fractal_dimension"] = clipped(
        values["artery_fractal_dimension"] + args.task3_artery_fd_offset,
        "artery_fractal_dimension",
    )
    calibrated["vein_fractal_dimension"] = clipped(
        values["vein_fractal_dimension"] + args.task3_vein_fd_offset,
        "vein_fractal_dimension",
    )
    # CRAE/CRVE are not scored, but keep the submitted triplet consistent.
    calibrated["CRAE"] = calibrated["AVR"] * calibrated["CRVE"]
    return calibrated


def process_task3(args: argparse.Namespace, destination: Path) -> dict[str, object]:
    before: dict[str, list[float]] = {key: [] for key in TASK3_KEYS}
    after: dict[str, list[float]] = {key: [] for key in TASK3_KEYS}

    destination.mkdir(parents=True)
    for stem in EXPECTED_STEMS:
        values = read_biomarkers(args.task3_source / f"{stem}.txt")
        calibrated = calibrate_biomarkers(values, args)
        for key in TASK3_KEYS:
            before[key].append(values[key])
            after[key].append(calibrated[key])

        with (destination / f"{stem}.txt").open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for key in TASK3_KEYS:
                handle.write(f"{key} {calibrated[key]:.6f}\n")

    summary = {}
    for key in TASK3_KEYS:
        before_array = np.asarray(before[key])
        after_array = np.asarray(after[key])
        summary[key] = {
            "mean_before": float(before_array.mean()),
            "mean_after": float(after_array.mean()),
            "mean_delta": float((after_array - before_array).mean()),
            "min_after": float(after_array.min()),
            "max_after": float(after_array.max()),
        }

    return {
        "source": str(args.task3_source.resolve()),
        "operation": "conservative_per_biomarker_calibration",
        "parameters": {
            "AVR_offset": args.task3_avr_offset,
            "artery_density_center": args.task3_artery_density_center,
            "artery_density_scale": args.task3_artery_density_scale,
            "artery_fractal_dimension_offset": args.task3_artery_fd_offset,
            "vein_fractal_dimension_offset": args.task3_vein_fd_offset,
            "vein_density": "unchanged",
            "CRAE": "recomputed as calibrated AVR * unchanged CRVE",
            "CRVE": "unchanged",
        },
        "summary": summary,
    }


def validate_output(output: Path) -> dict[str, object]:
    expected_pngs = {f"{stem}.png" for stem in EXPECTED_STEMS}
    expected_txts = {f"{stem}.txt" for stem in EXPECTED_STEMS}
    checks: dict[str, object] = {}

    for task in ("Task1", "Task2"):
        folder = output / task
        actual = {path.name for path in folder.glob("*.png")}
        if actual != expected_pngs:
            raise ValueError(f"{folder}: output filenames differ")
        for name in sorted(expected_pngs):
            with Image.open(folder / name) as image:
                if image.size != (1536, 1024) or image.mode != "RGB":
                    raise ValueError(
                        f"{folder / name}: size={image.size}, mode={image.mode}"
                    )
        checks[task] = {
            "count": len(actual),
            "first": min(actual),
            "last": max(actual),
            "image_size": [1536, 1024],
            "mode": "RGB",
        }

    task3_folder = output / "Task3"
    actual_txts = {path.name for path in task3_folder.glob("*.txt")}
    if actual_txts != expected_txts:
        raise ValueError(f"{task3_folder}: output filenames differ")
    for name in sorted(expected_txts):
        read_biomarkers(task3_folder / name)
    checks["Task3"] = {
        "count": len(actual_txts),
        "first": min(actual_txts),
        "last": max(actual_txts),
        "biomarker_keys": list(TASK3_KEYS),
    }
    return checks


def create_zip(output: Path) -> tuple[Path, str, int]:
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(
        zip_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for task in ("Task1", "Task2", "Task3"):
            for path in sorted((output / task).iterdir()):
                archive.write(path, arcname=f"{task}/{path.name}")

    expected_entries = {
        f"{task}/{stem}{suffix}"
        for task, suffix in (("Task1", ".png"), ("Task2", ".png"), ("Task3", ".txt"))
        for stem in EXPECTED_STEMS
    }
    with zipfile.ZipFile(zip_path) as archive:
        actual_entries = set(archive.namelist())
        if actual_entries != expected_entries:
            raise ValueError(
                f"ZIP entries differ; missing={expected_entries - actual_entries}, "
                f"extra={actual_entries - expected_entries}"
            )
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC validation failed: {bad_member}")

    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return zip_path, sha256, len(expected_entries)


def main() -> None:
    args = parse_args()
    validate_sources(args)

    args.output.mkdir(parents=True)
    task1_report = process_task1(args, args.output / "Task1")
    task2_report = process_task2(args, args.output / "Task2")
    task3_report = process_task3(args, args.output / "Task3")
    validation = validate_output(args.output)
    zip_path, zip_sha256, zip_entries = create_zip(args.output)

    manifest = {
        "candidate": args.output.name,
        "submission_directory": str(args.output.resolve()),
        "submission_zip": str(zip_path.resolve()),
        "submission_zip_sha256": zip_sha256,
        "zip_entries": zip_entries,
        "Task1": task1_report,
        "Task2": task2_report,
        "Task3": task3_report,
        "validation": validation,
    }
    manifest_path = args.output.with_name(f"{args.output.name}_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
