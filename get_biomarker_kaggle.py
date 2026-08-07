#!/usr/bin/env python3
"""CPU extraction of GAVE2 Task 3 biomarkers.

The biomarker definitions and TXT order follow Peng2004/CMRRWNet's
get_biomarker.py (Git blob d354391df5ecf44ab0f9d11974984d607c9621a8).

The official prediction script saves sigmoid probabilities.  The upstream
biomarker extractor, however, applies bitwise logic that only has a clear
meaning for binary RGB masks.  This version therefore thresholds R/G/B first
and then applies the upstream artery/vein logic.
"""

from __future__ import annotations

import argparse
import csv
import math
import traceback
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import medial_axis, skeletonize
from tqdm import tqdm


RESULT_KEYS = (
    "CRAE",
    "CRVE",
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)


def get_od_max_circle(od_mask: np.ndarray) -> tuple[tuple[int, int], float]:
    contours, _ = cv2.findContours(
        od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return (0, 0), 0.0

    max_contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(max_contour)
    return (int(cx), int(cy)), 2.0 * radius


def generate_annular_masks(
    av_img: np.ndarray, od_center: tuple[int, int], dd: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = av_img.shape[:2]
    cx, cy = od_center

    a_mask = np.zeros((height, width), dtype=np.uint8)
    b_mask = np.zeros((height, width), dtype=np.uint8)
    c_mask = np.zeros((height, width), dtype=np.uint8)

    od_radius = dd / 2.0
    a_outer_radius = od_radius + 0.5 * dd
    b_outer_radius = od_radius + 1.0 * dd
    c_outer_radius = od_radius + 2.0 * dd

    cv2.circle(a_mask, (cx, cy), int(a_outer_radius), 255, -1)
    cv2.circle(a_mask, (cx, cy), int(od_radius), 0, -1)

    cv2.circle(b_mask, (cx, cy), int(b_outer_radius), 255, -1)
    cv2.circle(b_mask, (cx, cy), int(a_outer_radius), 0, -1)

    cv2.circle(c_mask, (cx, cy), int(c_outer_radius), 255, -1)
    cv2.circle(c_mask, (cx, cy), int(b_outer_radius), 0, -1)

    return a_mask, b_mask, c_mask


def _medial_axis_with_distance(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use deterministic tie-breaking when supported by scikit-image."""
    try:
        return medial_axis(binary, return_distance=True, rng=0)
    except TypeError:
        return medial_axis(binary, return_distance=True)


def get_top_n_vessels_in_c(
    vessel_mask: np.ndarray, c_mask: np.ndarray, top_n: int = 6
) -> list[float]:
    """Match the upstream top-component diameter calculation.

    The upstream implementation constructs a full-image rectangular ROI for
    every selected component.  Cropping that same rectangle and adding a
    one-pixel zero border gives the same distance geometry with much less CPU
    and memory work.
    """
    vessel_in_c = cv2.bitwise_and(vessel_mask, vessel_mask, mask=c_mask)
    _, binary = cv2.threshold(vessel_in_c, 127, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8, ltype=cv2.CV_32S
    )

    vessels: list[tuple[int, int, int, int, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        vessels.append((-area, x, y, width, height))

    diameters: list[float] = []
    for _, x, y, width, height in sorted(vessels)[:top_n]:
        crop = vessel_mask[y : y + height, x : x + width] > 127
        crop = np.pad(crop, 1, mode="constant", constant_values=False)
        skeleton, distance = _medial_axis_with_distance(crop)
        values = distance[skeleton] * 2.0
        diameters.append(float(values.max()) if values.size else 0.0)

    diameters.extend([0.0] * (top_n - len(diameters)))
    return diameters


def calculate_crae_crve_revised(
    vessel_diameters: list[float], is_artery: bool
) -> float:
    coefficient = 0.88 if is_artery else 0.95
    values = sorted(vessel_diameters, reverse=True)

    while len(values) > 1:
        values = sorted(values, reverse=True)
        next_values: list[float] = []
        left = 0
        right = len(values) - 1

        while left < right:
            width_1 = values[left]
            width_2 = values[right]
            next_values.append(
                coefficient * math.sqrt(width_1**2 + width_2**2)
            )
            left += 1
            right -= 1

        values = next_values

    return float(values[0]) if values else 0.0


def calculate_density_in_c(vessel_mask: np.ndarray, c_mask: np.ndarray) -> float:
    _, vessel_binary = cv2.threshold(vessel_mask, 127, 1, cv2.THRESH_BINARY)
    _, c_binary = cv2.threshold(c_mask, 127, 1, cv2.THRESH_BINARY)
    c_pixels = int(np.sum(c_binary))
    if c_pixels == 0:
        return 0.0
    return float(np.sum(vessel_binary * c_binary) / c_pixels)


def _box_count_from_integral(integral: np.ndarray, box_size: int) -> int:
    """Count non-empty boxes using the same grid as the upstream loops."""
    height = integral.shape[0] - 1
    width = integral.shape[1] - 1
    row_start = np.arange(0, height, box_size, dtype=np.intp)
    col_start = np.arange(0, width, box_size, dtype=np.intp)
    row_end = np.minimum(row_start + box_size, height)
    col_end = np.minimum(col_start + box_size, width)

    sums = (
        integral[np.ix_(row_end, col_end)]
        - integral[np.ix_(row_start, col_end)]
        - integral[np.ix_(row_end, col_start)]
        + integral[np.ix_(row_start, col_start)]
    )
    return int(np.count_nonzero(sums))


def calculate_fractal_dimension_skeleton(binary_img: np.ndarray) -> float:
    """Vectorized equivalent of the baseline's all-box-size calculation."""
    if binary_img.max() == 0:
        return 0.0

    _, binary = cv2.threshold(binary_img, 127, 1, cv2.THRESH_BINARY)
    skeleton = skeletonize(binary).astype(np.uint8)
    rows, cols = skeleton.shape
    max_box_size = min(rows, cols) // 2
    integral = cv2.integral(skeleton, sdepth=cv2.CV_64F)

    log_inverse_sizes: list[float] = []
    log_counts: list[float] = []
    for box_size in range(1, max_box_size + 1):
        count = _box_count_from_integral(integral, box_size)
        if count > 0:
            log_inverse_sizes.append(math.log(1.0 / box_size))
            log_counts.append(math.log(count))

    if len(log_inverse_sizes) < 2:
        return 0.0

    return float(np.polyfit(log_inverse_sizes, log_counts, 1)[0])


def extract_av_masks(
    av_img_rgb: np.ndarray, probability_threshold: int
) -> tuple[np.ndarray, np.ndarray]:
    """Binarize probabilities, then reproduce the baseline channel logic.

    Input channels are R=artery, G=vessel, B=vein.  After thresholding, the
    upstream definitions are:
      artery = G AND NOT B
      vein   = G AND NOT R
    """
    red = av_img_rgb[:, :, 0]
    green = av_img_rgb[:, :, 1]
    blue = av_img_rgb[:, :, 2]

    red_binary = (red >= probability_threshold).astype(np.uint8)
    green_binary = (green >= probability_threshold).astype(np.uint8)
    blue_binary = (blue >= probability_threshold).astype(np.uint8)

    artery_mask = np.logical_and(green_binary, np.logical_not(blue_binary))
    vein_mask = np.logical_and(green_binary, np.logical_not(red_binary))
    return artery_mask.astype(np.uint8) * 255, vein_mask.astype(np.uint8) * 255


def process_av_indicators(
    av_dir: Path,
    disc_dir: Path,
    output_dir: Path,
    probability_threshold: int,
    expected_count: int,
    start_index: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    av_files = sorted(av_dir.glob("*.png"))
    if len(av_files) != expected_count:
        raise ValueError(
            f"Expected {expected_count} Task 2 PNGs, found {len(av_files)} in {av_dir}"
        )

    expected_names = [
        f"g_{index:03d}.png"
        for index in range(start_index, start_index + expected_count)
    ]
    actual_names = [path.name for path in av_files]
    if actual_names != expected_names:
        raise ValueError(
            "Task 2 names must be exactly "
            f"{expected_names[0]}..{expected_names[-1]}; got "
            f"{actual_names[0] if actual_names else 'none'}.."
            f"{actual_names[-1] if actual_names else 'none'}"
        )

    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for av_path in tqdm(av_files, desc="Calculating Task 3 biomarkers"):
        disc_path = disc_dir / av_path.name
        txt_path = output_dir / f"{av_path.stem}.txt"
        row: dict[str, object] = {
            "file": av_path.name,
            "threshold": probability_threshold,
            "status": "error",
            "error": "",
        }

        try:
            if not disc_path.exists():
                raise FileNotFoundError(f"Missing optic-disc mask: {disc_path.name}")

            av_bgr = cv2.imread(str(av_path), cv2.IMREAD_COLOR)
            disc_img = cv2.imread(str(disc_path), cv2.IMREAD_GRAYSCALE)
            if av_bgr is None:
                raise ValueError(f"Failed to read Task 2 PNG: {av_path}")
            if disc_img is None:
                raise ValueError(f"Failed to read optic-disc PNG: {disc_path}")

            av_img = cv2.cvtColor(av_bgr, cv2.COLOR_BGR2RGB)
            if av_img.shape[:2] != disc_img.shape:
                disc_img = cv2.resize(
                    disc_img, (av_img.shape[1], av_img.shape[0])
                )

            artery_mask, vein_mask = extract_av_masks(
                av_img, probability_threshold
            )
            _, od_binary = cv2.threshold(disc_img, 200, 255, cv2.THRESH_BINARY)
            od_center, disc_diameter = get_od_max_circle(od_binary)
            if disc_diameter <= 0:
                raise ValueError("No optic-disc contour was found")

            _, _, c_mask = generate_annular_masks(
                av_img, od_center, disc_diameter
            )
            artery_diameters = get_top_n_vessels_in_c(
                artery_mask, c_mask, top_n=6
            )
            vein_diameters = get_top_n_vessels_in_c(
                vein_mask, c_mask, top_n=6
            )

            crae = calculate_crae_crve_revised(artery_diameters, True)
            crve = calculate_crae_crve_revised(vein_diameters, False)
            if crae <= 0 or crve <= 0:
                raise ValueError(f"Non-positive CRAE/CRVE: CRAE={crae}, CRVE={crve}")

            results = {
                "CRAE": crae,
                "CRVE": crve,
                "AVR": crae / crve,
                "artery_density": calculate_density_in_c(artery_mask, c_mask),
                "vein_density": calculate_density_in_c(vein_mask, c_mask),
                "artery_fractal_dimension": calculate_fractal_dimension_skeleton(
                    artery_mask
                ),
                "vein_fractal_dimension": calculate_fractal_dimension_skeleton(
                    vein_mask
                ),
            }

            if not all(math.isfinite(float(value)) for value in results.values()):
                raise ValueError(f"Non-finite result: {results}")

            with txt_path.open("w", encoding="utf-8", newline="\n") as handle:
                for key in RESULT_KEYS:
                    handle.write(f"{key} {results[key]:.6f}\n")

            row.update(
                {
                    "status": "ok",
                    "od_center_x": od_center[0],
                    "od_center_y": od_center[1],
                    "od_diameter": disc_diameter,
                    "artery_top6_diameters": ";".join(
                        f"{value:.6f}" for value in artery_diameters
                    ),
                    "vein_top6_diameters": ";".join(
                        f"{value:.6f}" for value in vein_diameters
                    ),
                    **results,
                }
            )
        except Exception as exc:  # keep a complete QC report
            failures.append(av_path.name)
            row["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        rows.append(row)

    fieldnames = [
        "file",
        "threshold",
        "status",
        "error",
        "od_center_x",
        "od_center_y",
        "od_diameter",
        "artery_top6_diameters",
        "vein_top6_diameters",
        *RESULT_KEYS,
    ]
    with (output_dir / "task3_qc.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    txt_files = sorted(output_dir.glob("g_*.txt"))
    if failures or len(txt_files) != expected_count:
        raise RuntimeError(
            f"Task 3 failed for {failures}; generated {len(txt_files)}/{expected_count} TXT files. "
            f"See {output_dir / 'task3_qc.csv'}"
        )

    print(
        f"SUCCESS: generated {len(txt_files)} TXT files "
        f"({txt_files[0].name}..{txt_files[-1].name}) in {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--av-dir", type=Path, required=True)
    parser.add_argument("--disc-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Probability threshold on the saved 0..255 RGB channels (default: 127 ~= 0.5)",
    )
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First expected g_NNN index (default: 1; preliminary inference: 51)",
    )
    args = parser.parse_args()
    if not 1 <= args.threshold <= 254:
        parser.error("--threshold must be in 1..254")
    return args


def main() -> None:
    args = parse_args()
    process_av_indicators(
        args.av_dir.resolve(),
        args.disc_dir.resolve(),
        args.output_dir.resolve(),
        args.threshold,
        args.expected_count,
        args.start_index,
    )


if __name__ == "__main__":
    main()
