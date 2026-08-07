#!/usr/bin/env python3
"""Average probability PNGs produced by multiple baseline folds."""

from pathlib import Path
import argparse

import numpy as np
from PIL import Image


def pngs(folder):
    return {path.name: path for path in Path(folder).glob('*.png')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    inputs = [pngs(folder) for folder in args.inputs]
    expected = set(inputs[0])
    if not expected:
        raise ValueError(f'No PNG files found in {args.inputs[0]}')
    for folder, files in zip(args.inputs[1:], inputs[1:]):
        if set(files) != expected:
            missing = sorted(expected - set(files))
            extra = sorted(set(files) - expected)
            raise ValueError(
                f'{folder}: filenames differ; missing={missing}, extra={extra}'
            )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name in sorted(expected):
        arrays = []
        reference_shape = None
        for files in inputs:
            with Image.open(files[name]) as image:
                array = np.asarray(image.convert('RGB'), dtype=np.float32)
            if reference_shape is None:
                reference_shape = array.shape
            elif array.shape != reference_shape:
                raise ValueError(
                    f'{name}: shape {array.shape} != {reference_shape}'
                )
            arrays.append(array)

        averaged = np.rint(np.mean(arrays, axis=0)).astype(np.uint8)
        Image.fromarray(averaged, mode='RGB').save(output / name)

    print(
        f'Created {len(expected)} ensemble PNGs from '
        f'{len(inputs)} folds in {output}'
    )


if __name__ == '__main__':
    main()
