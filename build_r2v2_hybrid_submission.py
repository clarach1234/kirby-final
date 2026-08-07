#!/usr/bin/env python3
"""Build a GAVE2 submission with a specialist-channel Task 1 hybrid.

Task 1 uses the public R2-V2 AV model for artery/vein classification and the
kirby five-fold ensemble for binary-vessel segmentation:

    R = R2-V2 AV artery probability
    G = kirby vessel probability, with threshold 0.30 mapped to 0.50
    B = R2-V2 AV vein probability

Task 2 and Task 3 are copied byte-for-byte from an existing submission ZIP.
The script writes a JSON provenance/validation manifest beside the ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_STEMS = tuple(f"g_{index:03d}" for index in range(51, 101))
EXPECTED_SIZE = (1536, 1024)
TASK3_KEYS = {
    "CRAE",
    "CRVE",
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
}
R2V2_REPOSITORY = "https://github.com/j-morano/R2-V2"
R2V2_COMMIT = "7f6a8ea7a51782b1e0f89723a9ec137ba0a29913"
R2V2_AV_WEIGHT_SHA256 = (
    "74d425afb714384cb3f4d5db9cc852c1ea6d7552e46c866e29a3777db12b9d80"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r2v2-av-source",
        type=Path,
        default=Path("r2v2_validation_predictions/av"),
        help=(
            "R2-V2 AV predictions produced with --tta --use-gave-format. "
            "Their channels must be [artery, vessel, vein]."
        ),
    )
    parser.add_argument(
        "--kirby-task1-source",
        type=Path,
        default=Path("task1_01234_submission"),
        help="Raw equal five-fold kirby Task 1 probability PNGs.",
    )
    parser.add_argument(
        "--base-submission",
        type=Path,
        default=Path(
            "submission_experiments/"
            "kirby_exp_t1ch035040035_t2Db_endpoint_t3E_q035.zip"
        ),
        help="Submission ZIP supplying the unchanged Task 2 and Task 3 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "submission_experiments/"
            "kirby_hybrid_r2v2av_kirbyG030_t2Db_endpoint_t3E_q035"
        ),
        help="Output path without the .zip suffix.",
    )
    parser.add_argument(
        "--kirby-vessel-threshold",
        type=float,
        default=0.30,
        help="Original kirby G-channel threshold mapped to evaluator boundary 0.5.",
    )
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_exact_pngs(folder: Path) -> None:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing source directory: {folder}")
    expected = {f"{stem}.png" for stem in EXPECTED_STEMS}
    actual = {path.name for path in folder.glob("*.png")}
    if actual != expected:
        raise ValueError(
            f"{folder}: expected exactly 50 PNGs; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_rgb(array: np.ndarray) -> np.ndarray:
    if (array.shape[1], array.shape[0]) == EXPECTED_SIZE:
        return array
    image = Image.fromarray(array, mode="RGB")
    image = image.resize(EXPECTED_SIZE, resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def shift_probability_threshold(channel: np.ndarray, threshold: float) -> np.ndarray:
    """Map an original probability threshold to the evaluator's 0.5 boundary."""
    probability = channel.astype(np.float32) / 255.0
    numerator = probability * (1.0 - threshold)
    denominator = numerator + (1.0 - probability) * threshold
    shifted = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return np.rint(np.clip(shifted, 0.0, 1.0) * 255.0).astype(np.uint8)


def validate_base_submission(path: Path) -> dict[str, bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing base submission ZIP: {path}")
    expected = {
        *(f"Task1/{stem}.png" for stem in EXPECTED_STEMS),
        *(f"Task2/{stem}.png" for stem in EXPECTED_STEMS),
        *(f"Task3/{stem}.txt" for stem in EXPECTED_STEMS),
    }
    with zipfile.ZipFile(path) as archive:
        archive.testzip()
        actual = {name for name in archive.namelist() if not name.endswith("/")}
        if actual != expected:
            raise ValueError(
                f"{path}: expected exactly 150 task files; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        return {
            name: archive.read(name)
            for name in sorted(expected)
            if name.startswith(("Task2/", "Task3/"))
        }


def encode_png(array: np.ndarray) -> bytes:
    if array.shape != (EXPECTED_SIZE[1], EXPECTED_SIZE[0], 3):
        raise ValueError(f"Invalid hybrid shape: {array.shape}")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def validate_task3(content: bytes, name: str) -> None:
    values: dict[str, float] = {}
    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        normalized = line.replace(":", " ")
        key, raw_value = normalized.rsplit(maxsplit=1)
        values[key] = float(raw_value)
    if set(values) != TASK3_KEYS:
        raise ValueError(f"{name}: invalid biomarker keys: {sorted(values)}")
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"{name}: non-finite biomarker value")


def fixed_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 7, 31, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def main() -> None:
    args = parse_args()
    if not 0.0 < args.kirby_vessel_threshold < 1.0:
        raise ValueError("--kirby-vessel-threshold must be between 0 and 1")
    require_exact_pngs(args.r2v2_av_source)
    require_exact_pngs(args.kirby_task1_source)
    base_entries = validate_base_submission(args.base_submission)

    output_zip = args.output.with_suffix(".zip")
    manifest_path = args.output.with_name(f"{args.output.name}_manifest.json")
    for path in (output_zip, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    task1_entries: dict[str, bytes] = {}
    totals = {
        "artery_positive_at_0_5": 0,
        "vessel_positive_at_0_5": 0,
        "vein_positive_at_0_5": 0,
        "artery_outside_vessel": 0,
        "vein_outside_vessel": 0,
        "artery_vein_overlap": 0,
        "pixels": 0,
    }
    source_sizes: dict[str, int] = {}

    for stem in EXPECTED_STEMS:
        r2v2 = load_rgb(args.r2v2_av_source / f"{stem}.png")
        source_sizes[f"{r2v2.shape[1]}x{r2v2.shape[0]}"] = (
            source_sizes.get(f"{r2v2.shape[1]}x{r2v2.shape[0]}", 0) + 1
        )
        r2v2 = resize_rgb(r2v2)

        kirby = load_rgb(args.kirby_task1_source / f"{stem}.png")
        if (kirby.shape[1], kirby.shape[0]) != EXPECTED_SIZE:
            raise ValueError(
                f"{stem}: kirby image must be {EXPECTED_SIZE}, "
                f"got {(kirby.shape[1], kirby.shape[0])}"
            )
        vessel = shift_probability_threshold(
            kirby[:, :, 1],
            args.kirby_vessel_threshold,
        )
        hybrid = np.stack(
            (r2v2[:, :, 0], vessel, r2v2[:, :, 2]),
            axis=-1,
        )
        task1_entries[f"Task1/{stem}.png"] = encode_png(hybrid)

        artery_mask = hybrid[:, :, 0] >= 128
        vessel_mask = hybrid[:, :, 1] >= 128
        vein_mask = hybrid[:, :, 2] >= 128
        totals["artery_positive_at_0_5"] += int(artery_mask.sum())
        totals["vessel_positive_at_0_5"] += int(vessel_mask.sum())
        totals["vein_positive_at_0_5"] += int(vein_mask.sum())
        totals["artery_outside_vessel"] += int((artery_mask & ~vessel_mask).sum())
        totals["vein_outside_vessel"] += int((vein_mask & ~vessel_mask).sum())
        totals["artery_vein_overlap"] += int((artery_mask & vein_mask).sum())
        totals["pixels"] += int(vessel_mask.size)

    entries = {**task1_entries, **base_entries}
    expected_entry_count = len(EXPECTED_STEMS) * 3
    if len(entries) != expected_entry_count:
        raise RuntimeError(f"Expected 150 entries, got {len(entries)}")
    for name, content in entries.items():
        if name.startswith("Task3/"):
            validate_task3(content, name)

    with zipfile.ZipFile(
        output_zip,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name in sorted(entries):
            archive.writestr(fixed_zip_info(name), entries[name])

    with zipfile.ZipFile(output_zip) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")
        if len(archive.namelist()) != expected_entry_count:
            raise RuntimeError("Unexpected ZIP entry count after writing")

    vessel_positive = totals["vessel_positive_at_0_5"]
    manifest = {
        "candidate": args.output.name,
        "submission_zip": str(output_zip.resolve()),
        "submission_zip_sha256": sha256_path(output_zip),
        "zip_entries": expected_entry_count,
        "expected_stems": [EXPECTED_STEMS[0], EXPECTED_STEMS[-1]],
        "Task1": {
            "operation": "specialist_channel_hybrid",
            "channel_mapping": {
                "R_artery": "R2-V2 AV model artery probability",
                "G_vessel": (
                    "kirby equal five-fold vessel probability with original "
                    f"threshold {args.kirby_vessel_threshold:.2f} mapped to 0.50"
                ),
                "B_vein": "R2-V2 AV model vein probability",
            },
            "r2v2": {
                "repository": R2V2_REPOSITORY,
                "commit": R2V2_COMMIT,
                "model": "av",
                "inference": "--tta --use-gave-format",
                "weight_sha256": R2V2_AV_WEIGHT_SHA256,
                "prediction_source": str(args.r2v2_av_source.resolve()),
                "prediction_source_sizes": source_sizes,
                "resize_to_submission_if_needed": {
                    "size": list(EXPECTED_SIZE),
                    "method": "bilinear",
                },
            },
            "kirby_source": str(args.kirby_task1_source.resolve()),
            "statistics_at_evaluator_threshold_0_5": {
                **totals,
                "artery_outside_vessel_over_artery": (
                    totals["artery_outside_vessel"]
                    / totals["artery_positive_at_0_5"]
                ),
                "vein_outside_vessel_over_vein": (
                    totals["vein_outside_vessel"]
                    / totals["vein_positive_at_0_5"]
                ),
                "artery_vein_overlap_over_vessel": (
                    totals["artery_vein_overlap"] / vessel_positive
                ),
            },
        },
        "Task2": {
            "operation": "copied_byte_for_byte_from_base_submission",
            "source": str(args.base_submission.resolve()),
            "source_sha256": sha256_path(args.base_submission),
            "candidate": "Db + endpoint-only vein bridge",
            "known_leaderboard_score": 7.92935,
        },
        "Task3": {
            "operation": "copied_byte_for_byte_from_base_submission",
            "source": str(args.base_submission.resolve()),
            "source_sha256": sha256_path(args.base_submission),
            "candidate": "E + 35% rank-preserving quantile calibration",
            "known_leaderboard_score": 7.4347,
        },
        "validation": {
            "entry_count": expected_entry_count,
            "Task1": "50 RGB PNGs, 1536x1024",
            "Task2": "50 RGB PNGs copied unchanged",
            "Task3": "50 TXT files with exactly seven finite biomarkers",
            "zip_crc": "passed",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Created: {output_zip}")
    print(f"SHA-256: {manifest['submission_zip_sha256']}")
    print(f"Manifest: {manifest_path}")
    print(
        "Task1 positives (A/G/V): "
        f"{totals['artery_positive_at_0_5']:,}/"
        f"{totals['vessel_positive_at_0_5']:,}/"
        f"{totals['vein_positive_at_0_5']:,}"
    )
    print(
        "A/V outside G: "
        f"{totals['artery_outside_vessel']:,}/"
        f"{totals['vein_outside_vessel']:,}"
    )


if __name__ == "__main__":
    main()
