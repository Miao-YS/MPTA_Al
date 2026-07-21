#!/usr/bin/env python3
"""
Simplified 3D translation registration for sequential XCT TIFF slices.

This script:
1. Reads a reference TIFF stack and a moving TIFF stack.
2. Reconstructs each stack as a 3D volume.
3. Performs translation-only registration using Mattes mutual information.
4. Resamples the moving volume while preserving its original output grid.
5. Saves the registered volume as TIFF slices and writes the final transform.

Example
-------
python spatial_registration_simplified.py \
    --reference-dir data/reference \
    --moving-dir data/moving \
    --output-dir results/registered \
    --spacing 2.4464 2.4464 2.4464

Notes
-----
- The same physical unit must be used for all three spacing values.
- A fixed sampling seed is used to make metric sampling reproducible.
- This is a simplified, functionally equivalent research version intended
  for transparent reproduction of the spatial-registration procedure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import SimpleITK as sitk


def natural_sort_key(path: Path) -> list[object]:
    """Return a natural-sort key, e.g. slice2.tif before slice10.tif."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def collect_tiff_files(folder: Path) -> list[Path]:
    """Collect TIFF files from a folder in natural filename order."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    files = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ),
        key=natural_sort_key,
    )

    if not files:
        raise FileNotFoundError(f"No TIFF files were found in: {folder}")

    return files


def load_volume(
    folder: Path,
    spacing: Sequence[float],
) -> tuple[sitk.Image, list[Path]]:
    """Read a TIFF stack as a 3D SimpleITK image."""
    files = collect_tiff_files(folder)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in files])
    volume = reader.Execute()
    volume.SetSpacing(tuple(float(value) for value in spacing))

    if volume.GetDimension() != 3:
        raise ValueError(
            f"Expected a 3D volume, but obtained dimension "
            f"{volume.GetDimension()} from {folder}"
        )

    return volume, files


def register_translation(
    fixed: sitk.Image,
    moving: sitk.Image,
    *,
    shrink_factor: int = 2,
    histogram_bins: int = 50,
    sampling_percentage: float = 0.10,
    random_seed: int = 2026,
    learning_rate: float = 1.0,
    iterations: int = 100,
) -> tuple[sitk.Image, sitk.Transform, dict[str, object]]:
    """
    Register the moving volume to the fixed volume using translation only.

    The registered image is resampled on the original moving-image grid,
    preserving the moving stack dimensions, spacing, origin, and direction.
    """
    if shrink_factor < 1:
        raise ValueError("shrink_factor must be at least 1.")
    if not 0.0 < sampling_percentage <= 1.0:
        raise ValueError("sampling_percentage must be in the interval (0, 1].")

    fixed_float = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_float = sitk.Cast(moving, sitk.sitkFloat32)

    if shrink_factor > 1:
        factors = [shrink_factor] * 3
        fixed_work = sitk.Shrink(fixed_float, factors)
        moving_work = sitk.Shrink(moving_float, factors)
    else:
        fixed_work = fixed_float
        moving_work = moving_float

    initial_transform = sitk.TranslationTransform(3)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=histogram_bins
    )
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(
        sampling_percentage,
        random_seed,
    )
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed_work, moving_work)

    registered = sitk.Resample(
        moving,
        moving.GetSize(),
        final_transform,
        sitk.sitkLinear,
        moving.GetOrigin(),
        moving.GetSpacing(),
        moving.GetDirection(),
        0.0,
        moving.GetPixelID(),
    )

    summary = {
        "transform_type": final_transform.GetName(),
        "translation": [float(value) for value in final_transform.GetParameters()],
        "final_metric_value": float(registration.GetMetricValue()),
        "optimizer_stop_condition": registration.GetOptimizerStopConditionDescription(),
        "optimizer_iteration": int(registration.GetOptimizerIteration()),
        "shrink_factor": shrink_factor,
        "histogram_bins": histogram_bins,
        "sampling_percentage": sampling_percentage,
        "random_seed": random_seed,
        "learning_rate": learning_rate,
        "maximum_iterations": iterations,
    }

    return registered, final_transform, summary


def save_volume_as_slices(
    volume: sitk.Image,
    output_folder: Path,
    source_files: Sequence[Path],
) -> None:
    """Save a 3D volume as 2D TIFF slices using the moving-stack filenames."""
    output_folder.mkdir(parents=True, exist_ok=True)

    number_of_slices = volume.GetSize()[2]
    for z_index in range(number_of_slices):
        slice_image = volume[:, :, z_index]

        if z_index < len(source_files):
            filename = source_files[z_index].name
        else:
            filename = f"slice_{z_index:04d}.tif"

        sitk.WriteImage(slice_image, str(output_folder / filename))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perform translation-only registration between two 3D XCT TIFF stacks."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Folder containing the reference TIFF slices.",
    )
    parser.add_argument(
        "--moving-dir",
        type=Path,
        required=True,
        help="Folder containing the moving TIFF slices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder in which registered TIFF slices will be saved.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=(2.4464, 2.4464, 2.4464),
        help="Voxel spacing in x, y, and z. Default: 2.4464 2.4464 2.4464.",
    )
    parser.add_argument(
        "--shrink-factor",
        type=int,
        default=2,
        help="Integer downsampling factor used during registration. Default: 2.",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=50,
        help="Number of Mattes mutual-information histogram bins. Default: 50.",
    )
    parser.add_argument(
        "--sampling-percentage",
        type=float,
        default=0.10,
        help="Fraction of voxels sampled for metric evaluation. Default: 0.10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Fixed random seed for reproducible metric sampling. Default: 2026.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0,
        help="Gradient-descent learning rate. Default: 1.0.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Maximum number of optimizer iterations. Default: 100.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    try:
        print("[1/4] Reading reference volume...")
        fixed, fixed_files = load_volume(args.reference_dir, args.spacing)
        print(
            f"      Loaded {len(fixed_files)} slices; "
            f"volume size = {fixed.GetSize()}."
        )

        print("[2/4] Reading moving volume...")
        moving, moving_files = load_volume(args.moving_dir, args.spacing)
        print(
            f"      Loaded {len(moving_files)} slices; "
            f"volume size = {moving.GetSize()}."
        )

        print("[3/4] Performing 3D translation registration...")
        registered, final_transform, summary = register_translation(
            fixed,
            moving,
            shrink_factor=args.shrink_factor,
            histogram_bins=args.histogram_bins,
            sampling_percentage=args.sampling_percentage,
            random_seed=args.seed,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(
            final_transform,
            str(args.output_dir / "translation_transform.tfm"),
        )

        run_summary = {
            "reference_directory": str(args.reference_dir),
            "moving_directory": str(args.moving_dir),
            "output_directory": str(args.output_dir),
            "reference_slice_count": len(fixed_files),
            "moving_slice_count": len(moving_files),
            "reference_size": list(fixed.GetSize()),
            "moving_size": list(moving.GetSize()),
            "voxel_spacing": list(args.spacing),
            **summary,
        }
        with (args.output_dir / "registration_summary.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(run_summary, file, indent=2, ensure_ascii=False)

        print(
            "      Final translation = "
            f"{tuple(round(value, 6) for value in summary['translation'])}"
        )
        print(f"      Final metric value = {summary['final_metric_value']:.6g}")

        print("[4/4] Saving registered TIFF slices...")
        save_volume_as_slices(registered, args.output_dir, moving_files)

        print(f"Registration completed. Results: {args.output_dir}")
        return 0

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
