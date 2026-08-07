#!/usr/bin/env python3
"""Build a three-task GAVE2 exploration submission.

The candidate changes exactly one idea inside each task:

* Task 1: map channel thresholds A/G/V = 0.35/0.40/0.35 to the evaluator's
  0.50 boundary with a probability-preserving logit shift.
* Task 2: start from Db and add only short, directionally plausible bridges
  between endpoints of different vein components. Existing probabilities are
  never reduced.
* Task 3: shrink E's empirical biomarker quantiles 35% toward the labeled
  training distribution. The mapping is rank-preserving and uses no
  validation labels.

Only Task1/, Task2/, and Task3/ are stored in the submission ZIP. A detailed
manifest is written next to the ZIP.
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
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from skimage.draw import line
from skimage.morphology import skeletonize


EXPECTED_STEMS = tuple(f"g_{index:03d}" for index in range(51, 101))
TRAIN_STEMS = tuple(f"g_{index:03d}" for index in range(1, 51))
TASK3_KEYS = (
    "CRAE",
    "CRVE",
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)
SCORED_TASK3_KEYS = (
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)


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
        default=Path("task2_Db_submission"),
    )
    parser.add_argument(
        "--task3-source",
        type=Path,
        default=Path("task3_E_submission"),
    )
    parser.add_argument(
        "--task3-training-labels",
        type=Path,
        default=Path("data/training/biomarker"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "submission_experiments/"
            "kirby_exp_t1ch035040035_t2Db_endpoint_t3E_q035"
        ),
    )
    parser.add_argument(
        "--task1-thresholds",
        type=float,
        nargs=3,
        metavar=("A", "G", "V"),
        default=(0.35, 0.40, 0.35),
    )
    parser.add_argument("--task2-core-threshold", type=float, default=0.50)
    parser.add_argument("--task2-min-gap", type=float, default=2.0)
    parser.add_argument("--task2-max-gap", type=float, default=8.0)
    parser.add_argument("--task2-min-component", type=int, default=3)
    parser.add_argument("--task2-min-large-component", type=int, default=20)
    parser.add_argument("--task2-min-alignment", type=float, default=0.25)
    parser.add_argument("--task2-min-mean-vein", type=float, default=0.12)
    parser.add_argument("--task2-min-mean-vessel", type=float, default=0.25)
    parser.add_argument("--task2-max-red-excess", type=float, default=0.18)
    parser.add_argument("--task2-added-confidence", type=float, default=0.60)
    parser.add_argument("--task2-max-added-ratio", type=float, default=0.0075)
    parser.add_argument("--task2-max-bridges-per-image", type=int, default=64)
    parser.add_argument(
        "--task2-mode",
        choices=("endpoint", "passthrough"),
        default="endpoint",
    )
    parser.add_argument(
        "--task3-mode",
        choices=("quantile", "selective_fd"),
        default="quantile",
    )
    parser.add_argument("--task3-quantile-strength", type=float, default=0.35)
    parser.add_argument("--task3-artery-fd-offset", type=float, default=-0.013)
    parser.add_argument("--task3-vein-fd-offset", type=float, default=0.012)
    return parser.parse_args()


def require_probability(value: float, label: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{label} must be strictly between 0 and 1: {value}")


def expected_names(stems: tuple[str, ...], suffix: str) -> set[str]:
    return {f"{stem}{suffix}" for stem in stems}


def require_exact_files(
    folder: Path,
    stems: tuple[str, ...],
    suffix: str,
) -> None:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing source directory: {folder}")
    expected = expected_names(stems, suffix)
    actual = {path.name for path in folder.glob(f"*{suffix}")}
    if actual != expected:
        raise ValueError(
            f"{folder}: expected exactly {len(expected)} files; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def validate_sources(args: argparse.Namespace) -> None:
    for index, threshold in enumerate(args.task1_thresholds):
        require_probability(threshold, f"task1_thresholds[{index}]")
    for value, label in (
        (args.task2_core_threshold, "task2_core_threshold"),
        (args.task2_min_mean_vein, "task2_min_mean_vein"),
        (args.task2_min_mean_vessel, "task2_min_mean_vessel"),
        (args.task2_added_confidence, "task2_added_confidence"),
        (args.task2_max_added_ratio, "task2_max_added_ratio"),
        (args.task3_quantile_strength, "task3_quantile_strength"),
    ):
        require_probability(value, label)

    if not 0.0 <= args.task2_min_alignment <= 1.0:
        raise ValueError("task2_min_alignment must be in [0, 1]")
    if not 0.0 <= args.task2_max_red_excess <= 1.0:
        raise ValueError("task2_max_red_excess must be in [0, 1]")
    if not 0.0 < args.task2_min_gap <= args.task2_max_gap:
        raise ValueError("Task 2 gap limits are invalid")
    if args.task2_min_component < 1:
        raise ValueError("task2_min_component must be positive")
    if args.task2_min_large_component < args.task2_min_component:
        raise ValueError(
            "task2_min_large_component must be >= task2_min_component"
        )
    if args.task2_max_bridges_per_image < 1:
        raise ValueError("task2_max_bridges_per_image must be positive")
    if args.task2_added_confidence <= 0.5:
        raise ValueError("Task 2 added confidence must exceed 0.5")

    require_exact_files(args.task1_source, EXPECTED_STEMS, ".png")
    require_exact_files(args.task2_source, EXPECTED_STEMS, ".png")
    require_exact_files(args.task3_source, EXPECTED_STEMS, ".txt")
    require_exact_files(args.task3_training_labels, TRAIN_STEMS, ".txt")

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {args.output}"
        )
    zip_path = args.output.with_suffix(".zip")
    manifest_path = args.output.with_name(f"{args.output.name}_manifest.json")
    if zip_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: "
            f"{zip_path} or {manifest_path}"
        )


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (1536, 1024):
            raise ValueError(f"{path}: expected 1536x1024, got {image.size}")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb(array: np.ndarray, path: Path) -> None:
    if array.shape != (1024, 1536, 3) or array.dtype != np.uint8:
        raise ValueError(
            f"Invalid output array for {path}: "
            f"shape={array.shape}, dtype={array.dtype}"
        )
    Image.fromarray(array, mode="RGB").save(path)


def shift_probability_threshold(
    array: np.ndarray,
    thresholds: tuple[float, float, float],
) -> np.ndarray:
    """Map three original channel thresholds to 0.5 via logit shifts."""
    probability = array.astype(np.float32) / 255.0
    threshold = np.asarray(thresholds, dtype=np.float32).reshape(1, 1, 3)
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
    thresholds = tuple(float(value) for value in args.task1_thresholds)
    before_positive = np.zeros(3, dtype=np.int64)
    after_positive = np.zeros(3, dtype=np.int64)

    destination.mkdir(parents=True)
    for stem in EXPECTED_STEMS:
        source = load_rgb(args.task1_source / f"{stem}.png")
        shifted = shift_probability_threshold(source, thresholds)
        before_positive += np.sum(source >= 128, axis=(0, 1))
        after_positive += np.sum(shifted >= 128, axis=(0, 1))
        save_rgb(shifted, destination / f"{stem}.png")

    return {
        "source": str(args.task1_source.resolve()),
        "operation": "channelwise_logit_threshold_shift",
        "original_thresholds_mapped_to_0_5": dict(
            zip(channel_names, thresholds)
        ),
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


def endpoint_direction(
    endpoint: np.ndarray,
    component: int,
    skeleton: np.ndarray,
    labels: np.ndarray,
    radius: int = 4,
) -> np.ndarray | None:
    """Estimate the outward direction at a skeleton endpoint."""
    row, col = (int(endpoint[0]), int(endpoint[1]))
    r0 = max(0, row - radius)
    r1 = min(skeleton.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(skeleton.shape[1], col + radius + 1)
    local = np.argwhere(
        skeleton[r0:r1, c0:c1]
        & (labels[r0:r1, c0:c1] == component)
    )
    if len(local) < 2:
        return None
    local = local + np.array([r0, c0])
    distances = np.linalg.norm(local - endpoint, axis=1)
    interior = local[distances > 0.5]
    if not len(interior):
        return None
    centroid = interior.mean(axis=0)
    outward = endpoint.astype(np.float64) - centroid
    norm = np.linalg.norm(outward)
    if norm <= 1e-8:
        return None
    return outward / norm


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size + 1, dtype=np.int32)

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        while parent != value:
            grandparent = int(self.parent[parent])
            self.parent[value] = grandparent
            value = parent
            parent = grandparent
        return value

    def union(self, first: int, second: int) -> bool:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return False
        self.parent[root_second] = root_first
        return True


def endpoint_bridge_vein(
    image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    """Add short endpoint-to-endpoint vein bridges without deleting pixels."""
    probability = image.astype(np.float32) / 255.0
    artery = probability[:, :, 0]
    vessel = probability[:, :, 1]
    vein = probability[:, :, 2]

    core = vein >= args.task2_core_threshold
    labels, component_count = ndi.label(
        core,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    component_sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    skeleton = skeletonize(core)
    neighbors = ndi.convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    ) - skeleton.astype(np.uint8)
    endpoints = np.argwhere(skeleton & (neighbors == 1))

    endpoint_records: list[tuple[np.ndarray, int, np.ndarray]] = []
    for endpoint in endpoints:
        component = int(labels[tuple(endpoint)])
        if component <= 0:
            continue
        if component_sizes[component] < args.task2_min_component:
            continue
        direction = endpoint_direction(
            endpoint,
            component,
            skeleton,
            labels,
        )
        if direction is not None:
            endpoint_records.append((endpoint, component, direction))

    candidates: list[
        tuple[float, int, int, int, int, np.ndarray, np.ndarray]
    ] = []
    if len(endpoint_records) >= 2:
        coordinates = np.asarray(
            [record[0] for record in endpoint_records],
            dtype=np.float64,
        )
        tree = cKDTree(coordinates)
        for first, second in tree.query_pairs(args.task2_max_gap):
            endpoint_1, component_1, direction_1 = endpoint_records[first]
            endpoint_2, component_2, direction_2 = endpoint_records[second]
            if component_1 == component_2:
                continue
            if (
                component_sizes[component_1] < args.task2_min_large_component
                and component_sizes[component_2] < args.task2_min_large_component
            ):
                continue

            difference = endpoint_2.astype(np.float64) - endpoint_1
            distance = float(np.linalg.norm(difference))
            if not args.task2_min_gap <= distance <= args.task2_max_gap:
                continue
            toward_second = difference / distance
            alignment_1 = float(np.dot(direction_1, toward_second))
            alignment_2 = float(np.dot(direction_2, -toward_second))
            if min(alignment_1, alignment_2) < args.task2_min_alignment:
                continue

            rr, cc = line(
                int(endpoint_1[0]),
                int(endpoint_1[1]),
                int(endpoint_2[0]),
                int(endpoint_2[1]),
            )
            gap = ~core[rr, cc]
            if not gap.any():
                continue
            gap_rr = rr[gap]
            gap_cc = cc[gap]
            mean_vein = float(vein[gap_rr, gap_cc].mean())
            mean_vessel = float(vessel[gap_rr, gap_cc].mean())
            mean_red_excess = float(
                np.maximum(
                    artery[gap_rr, gap_cc] - vein[gap_rr, gap_cc],
                    0.0,
                ).mean()
            )
            if mean_vein < args.task2_min_mean_vein:
                continue
            if mean_vessel < args.task2_min_mean_vessel:
                continue
            if mean_red_excess > args.task2_max_red_excess:
                continue

            score = (
                0.55 * mean_vein
                + 0.25 * mean_vessel
                + 0.10 * min(alignment_1, alignment_2)
                - 0.35 * mean_red_excess
                - 0.01 * distance
            )
            candidates.append(
                (
                    score,
                    first,
                    second,
                    component_1,
                    component_2,
                    gap_rr,
                    gap_cc,
                )
            )

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = np.zeros(core.shape, dtype=bool)
    disjoint_set = DisjointSet(component_count)
    max_added = max(
        1,
        int(math.floor(core.sum() * args.task2_max_added_ratio)),
    )
    bridge_count = 0
    candidate_added_pixels = 0
    for (
        _score,
        _first,
        _second,
        component_1,
        component_2,
        gap_rr,
        gap_cc,
    ) in candidates:
        if bridge_count >= args.task2_max_bridges_per_image:
            break
        if disjoint_set.find(component_1) == disjoint_set.find(component_2):
            continue
        new_pixel = ~selected[gap_rr, gap_cc]
        new_count = int(new_pixel.sum())
        if new_count == 0:
            continue
        if int(selected.sum()) + new_count > max_added:
            continue
        selected[gap_rr[new_pixel], gap_cc[new_pixel]] = True
        disjoint_set.union(component_1, component_2)
        bridge_count += 1
        candidate_added_pixels += new_count

    result = probability.copy()
    result[:, :, 2][selected] = np.maximum(
        result[:, :, 2][selected],
        args.task2_added_confidence,
    )
    result[:, :, 1][selected] = np.maximum(
        result[:, :, 1][selected],
        result[:, :, 2][selected],
    )
    output = np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)

    changed = np.any(output != image, axis=2)
    reduced = bool(np.any(output.astype(np.int16) < image.astype(np.int16)))
    return output, {
        "core_pixels": int(core.sum()),
        "components": int(component_count),
        "endpoints": int(len(endpoint_records)),
        "eligible_candidate_bridges": int(len(candidates)),
        "selected_bridges": int(bridge_count),
        "candidate_added_pixels": int(candidate_added_pixels),
        "changed_pixels": int(changed.sum()),
        "max_added_pixels": int(max_added),
        "probability_reduced": reduced,
    }


def process_task2(args: argparse.Namespace, destination: Path) -> dict[str, object]:
    total_keys = (
        "core_pixels",
        "components",
        "endpoints",
        "eligible_candidate_bridges",
        "selected_bridges",
        "candidate_added_pixels",
        "changed_pixels",
        "max_added_pixels",
    )
    totals = {key: 0 for key in total_keys}
    per_image: dict[str, dict[str, object]] = {}

    destination.mkdir(parents=True)
    for stem in EXPECTED_STEMS:
        source = load_rgb(args.task2_source / f"{stem}.png")
        if args.task2_mode == "endpoint":
            recovered, stats = endpoint_bridge_vein(source, args)
        else:
            recovered = source.copy()
            core_pixels = int(
                (source[:, :, 2] >= round(255 * args.task2_core_threshold)).sum()
            )
            stats = {
                "core_pixels": core_pixels,
                "components": 0,
                "endpoints": 0,
                "eligible_candidate_bridges": 0,
                "selected_bridges": 0,
                "candidate_added_pixels": 0,
                "changed_pixels": 0,
                "max_added_pixels": 0,
                "probability_reduced": False,
            }
        save_rgb(recovered, destination / f"{stem}.png")
        for key in total_keys:
            totals[key] += int(stats[key])
        if stats["probability_reduced"]:
            raise ValueError(f"{stem}: Task 2 unexpectedly reduced probabilities")
        per_image[stem] = stats

    return {
        "source": str(args.task2_source.resolve()),
        "operation": (
            "additive_endpoint_only_vein_bridge"
            if args.task2_mode == "endpoint"
            else "passthrough"
        ),
        "parameters": {
            "core_threshold": args.task2_core_threshold,
            "gap_pixels": [args.task2_min_gap, args.task2_max_gap],
            "min_component_pixels": args.task2_min_component,
            "min_large_component_pixels": args.task2_min_large_component,
            "min_direction_alignment": args.task2_min_alignment,
            "min_gap_mean_vein_probability": args.task2_min_mean_vein,
            "min_gap_mean_vessel_probability": args.task2_min_mean_vessel,
            "max_gap_mean_red_excess": args.task2_max_red_excess,
            "added_confidence": args.task2_added_confidence,
            "max_added_over_core_ratio_per_image": args.task2_max_added_ratio,
            "max_bridges_per_image": args.task2_max_bridges_per_image,
        },
        "existing_probabilities_reduced": False,
        "totals": totals,
        "changed_over_core_ratio": (
            float(totals["changed_pixels"] / totals["core_pixels"])
            if totals["core_pixels"]
            else 0.0
        ),
        "per_image": per_image,
    }


def read_biomarkers(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}: malformed line: {raw_line!r}")
        key, raw_value = parts
        if key in values:
            raise ValueError(f"{path}: duplicate biomarker: {key}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite biomarker: {raw_line!r}")
        values[key] = value
    if set(values) != set(TASK3_KEYS):
        raise ValueError(
            f"{path}: biomarker keys differ; "
            f"missing={set(TASK3_KEYS) - set(values)}, "
            f"extra={set(values) - set(TASK3_KEYS)}"
        )
    return values


def read_biomarker_matrix(
    folder: Path,
    stems: tuple[str, ...],
) -> dict[str, np.ndarray]:
    rows = [read_biomarkers(folder / f"{stem}.txt") for stem in stems]
    return {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in TASK3_KEYS
    }


def empirical_quantile_targets(
    predictions: np.ndarray,
    training_labels: np.ndarray,
) -> np.ndarray:
    """Map prediction ranks to the empirical training-label quantiles."""
    quantiles = (rankdata(predictions, method="average") - 0.5) / len(predictions)
    try:
        return np.quantile(training_labels, quantiles, method="linear")
    except TypeError:
        return np.quantile(training_labels, quantiles, interpolation="linear")


def process_task3(args: argparse.Namespace, destination: Path) -> dict[str, object]:
    source = read_biomarker_matrix(args.task3_source, EXPECTED_STEMS)
    training = read_biomarker_matrix(args.task3_training_labels, TRAIN_STEMS)
    calibrated = {key: values.copy() for key, values in source.items()}
    quantile_targets: dict[str, np.ndarray] = {}

    strength = float(args.task3_quantile_strength)
    if args.task3_mode == "quantile":
        for key in SCORED_TASK3_KEYS:
            target = empirical_quantile_targets(source[key], training[key])
            quantile_targets[key] = target
            calibrated[key] = (1.0 - strength) * source[key] + strength * target
    else:
        calibrated["artery_fractal_dimension"] = (
            source["artery_fractal_dimension"] + args.task3_artery_fd_offset
        )
        calibrated["vein_fractal_dimension"] = (
            source["vein_fractal_dimension"] + args.task3_vein_fd_offset
        )

    # CRAE/CRVE are not displayed in the official Task 3 metric breakdown.
    # Keep CRVE unchanged and maintain CRAE = AVR * CRVE consistency.
    calibrated["CRVE"] = source["CRVE"].copy()
    calibrated["CRAE"] = calibrated["AVR"] * calibrated["CRVE"]

    destination.mkdir(parents=True)
    for index, stem in enumerate(EXPECTED_STEMS):
        with (destination / f"{stem}.txt").open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for key in TASK3_KEYS:
                value = float(calibrated[key][index])
                if not math.isfinite(value):
                    raise ValueError(f"{stem}: non-finite calibrated {key}")
                handle.write(f"{key} {value:.6f}\n")

    summary: dict[str, object] = {}
    for key in TASK3_KEYS:
        before = source[key]
        after = calibrated[key]
        item: dict[str, object] = {
            "training_mean": float(training[key].mean()),
            "training_std": float(training[key].std()),
            "mean_before": float(before.mean()),
            "std_before": float(before.std()),
            "mean_after": float(after.mean()),
            "std_after": float(after.std()),
            "mean_delta": float((after - before).mean()),
            "min_after": float(after.min()),
            "max_after": float(after.max()),
        }
        if key in quantile_targets:
            item["mean_quantile_target"] = float(quantile_targets[key].mean())
        summary[key] = item

    return {
        "source": str(args.task3_source.resolve()),
        "training_distribution": str(args.task3_training_labels.resolve()),
        "operation": (
            "rank_preserving_empirical_quantile_shrinkage"
            if args.task3_mode == "quantile"
            else "selective_fractal_dimension_offsets"
        ),
        "quantile_strength": (
            strength if args.task3_mode == "quantile" else None
        ),
        "scored_keys_calibrated": (
            list(SCORED_TASK3_KEYS)
            if args.task3_mode == "quantile"
            else [
                "artery_fractal_dimension",
                "vein_fractal_dimension",
            ]
        ),
        "selective_offsets": (
            None
            if args.task3_mode == "quantile"
            else {
                "artery_fractal_dimension": args.task3_artery_fd_offset,
                "vein_fractal_dimension": args.task3_vein_fd_offset,
            }
        ),
        "validation_labels_used": False,
        "CRAE": "recomputed as calibrated AVR * unchanged CRVE",
        "CRVE": "unchanged",
        "summary": summary,
    }


def validate_output(output: Path) -> dict[str, object]:
    expected_pngs = expected_names(EXPECTED_STEMS, ".png")
    expected_txts = expected_names(EXPECTED_STEMS, ".txt")
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
        for task, suffix in (
            ("Task1", ".png"),
            ("Task2", ".png"),
            ("Task3", ".txt"),
        )
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
