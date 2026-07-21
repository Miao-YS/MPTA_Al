#!/usr/bin/env python3
"""
Simplified micropore-parameter extraction from a binary XCT TIFF stack.

Core workflow
-------------
1. Read sequential TIFF slices and reconstruct a 3D volume.
2. Convert the volume to a binary pore mask using: intensity > threshold.
3. Identify individual micropores using 26-neighbour connectivity.
4. Exclude connected regions smaller than the specified voxel threshold.
5. Calculate the principal micropore parameters and export them to CSV.

Calculated parameters
---------------------
- Centroid coordinates
- Voxel count and physical volume
- Equivalent diameter
- Number of occupied slices and z-direction thickness
- Principal-axis aspect ratio
- Normalized principal variances (Lambda1 and Lambda2)

Coordinate convention
---------------------
To remain compatible with the original analysis:
- X and Y pixel indices are zero-based.
- Z is reported as a one-based slice coordinate.
- Physical coordinates are calculated using the supplied voxel size.

This script is a simplified, functionally equivalent version intended for
transparent reproduction and public code sharing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


OUTPUT_COLUMNS = [
    "GlobalID",
    "Group",
    "RegionID",
    "Centroid_idx_X",
    "Centroid_idx_Y",
    "Centroid_idx_Z",
    "Centroid_X_um",
    "Centroid_Y_um",
    "Centroid_Z_um",
    "VoxelCount",
    "Volume_um3",
    "EquivalentDiameter_um",
    "SliceCount",
    "Z_thickness_um",
    "AspectRatio_short_long",
    "Lambda1",
    "Lambda2",
]


def natural_sort_key(path: Path) -> list[object]:
    """Sort filenames naturally, e.g. slice2.tif before slice10.tif."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def collect_tiff_files(folder: Path) -> list[Path]:
    """Return TIFF files in natural filename order."""
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
        raise FileNotFoundError(f"No TIFF slices were found in: {folder}")

    return files


def load_tiff_stack(folder: Path) -> tuple[np.ndarray, list[Path]]:
    """Load a sequence of equally sized TIFF slices as a 3D array (z, y, x)."""
    files = collect_tiff_files(folder)
    slices: list[np.ndarray] = []

    expected_shape: tuple[int, int] | None = None
    for path in files:
        with Image.open(path) as image:
            array = np.asarray(image)

        if array.ndim == 3:
            # Use the first channel if an RGB/RGBA image is supplied.
            array = array[..., 0]

        if array.ndim != 2:
            raise ValueError(f"Slice is not two-dimensional: {path}")

        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError(
                f"Inconsistent slice dimensions: {path.name} has shape "
                f"{array.shape}, expected {expected_shape}."
            )

        slices.append(array)

    return np.stack(slices, axis=0), files


def principal_axis_features(
    centered_coordinates_um: np.ndarray,
) -> tuple[float, float, float]:
    """
    Calculate a short-to-long principal-axis ratio and normalized eigenvalues.

    The calculation is based on the covariance matrix of physical voxel
    coordinates. NaN values are returned when the region is too small or
    geometrically degenerate.
    """
    if centered_coordinates_um.shape[0] < 3:
        return math.nan, math.nan, math.nan

    covariance = np.cov(centered_coordinates_um.T)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)

    total = float(eigenvalues.sum())
    if total <= 0.0 or eigenvalues[0] <= 1e-12:
        return math.nan, math.nan, math.nan

    principal_radii = np.sqrt(eigenvalues)
    aspect_ratio = float(principal_radii[-1] / principal_radii[0])
    normalized = eigenvalues / total

    return aspect_ratio, float(normalized[0]), float(normalized[1])


def measure_region(
    region_id: int,
    coordinates_zyx: np.ndarray,
    voxel_size_xyz: Sequence[float],
    group_name: str,
) -> dict[str, object]:
    """Calculate micropore parameters for one connected region."""
    sx, sy, sz = (float(value) for value in voxel_size_xyz)

    # Original convention: x/y indices are zero-based; z is one-based.
    z_slice = coordinates_zyx[:, 0].astype(np.float64) + 1.0
    y_index = coordinates_zyx[:, 1].astype(np.float64)
    x_index = coordinates_zyx[:, 2].astype(np.float64)

    coordinates_index_zyx = np.column_stack((z_slice, y_index, x_index))
    coordinates_um_zyx = np.column_stack(
        (z_slice * sz, y_index * sy, x_index * sx)
    )

    centroid_index_zyx = coordinates_index_zyx.mean(axis=0)
    centroid_um_zyx = coordinates_um_zyx.mean(axis=0)
    centered_um = coordinates_um_zyx - centroid_um_zyx

    aspect_ratio, lambda1, lambda2 = principal_axis_features(centered_um)

    voxel_count = int(coordinates_zyx.shape[0])
    voxel_volume_um3 = sx * sy * sz
    volume_um3 = voxel_count * voxel_volume_um3
    equivalent_diameter_um = (
        (6.0 * volume_um3 / math.pi) ** (1.0 / 3.0)
        if volume_um3 > 0.0
        else 0.0
    )

    unique_slices = np.unique(z_slice)
    slice_count = int(unique_slices.size)
    z_thickness_um = (
        float(unique_slices.max() - unique_slices.min() + 1.0) * sz
    )

    return {
        "GlobalID": f"{group_name}-{region_id}",
        "Group": group_name,
        "RegionID": region_id,
        "Centroid_idx_X": float(centroid_index_zyx[2]),
        "Centroid_idx_Y": float(centroid_index_zyx[1]),
        "Centroid_idx_Z": float(centroid_index_zyx[0]),
        "Centroid_X_um": float(centroid_um_zyx[2]),
        "Centroid_Y_um": float(centroid_um_zyx[1]),
        "Centroid_Z_um": float(centroid_um_zyx[0]),
        "VoxelCount": voxel_count,
        "Volume_um3": float(volume_um3),
        "EquivalentDiameter_um": float(equivalent_diameter_um),
        "SliceCount": slice_count,
        "Z_thickness_um": float(z_thickness_um),
        "AspectRatio_short_long": aspect_ratio,
        "Lambda1": lambda1,
        "Lambda2": lambda2,
    }


def extract_micropore_parameters(
    volume: np.ndarray,
    *,
    threshold: float,
    min_voxels: int,
    voxel_size_xyz: Sequence[float],
    group_name: str,
) -> tuple[list[dict[str, object]], int]:
    """
    Label a binary pore volume and calculate parameters for valid regions.

    Pore voxels are defined as values strictly greater than ``threshold``.
    A full 3 x 3 x 3 structure is used, corresponding to 26-neighbour
    connectivity in three dimensions.
    """
    if min_voxels < 1:
        raise ValueError("min_voxels must be at least 1.")
    if any(float(value) <= 0.0 for value in voxel_size_xyz):
        raise ValueError("All voxel-size values must be positive.")

    pore_mask = volume > threshold
    connectivity_26 = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_volume, number_of_regions = ndimage.label(
        pore_mask,
        structure=connectivity_26,
    )

    results: list[dict[str, object]] = []

    # find_objects limits each calculation to the bounding box of one region.
    region_boxes = ndimage.find_objects(labeled_volume)
    for region_id, region_box in enumerate(region_boxes, start=1):
        if region_box is None:
            continue

        local_labels = labeled_volume[region_box]
        local_coordinates = np.argwhere(local_labels == region_id)
        if local_coordinates.shape[0] < min_voxels:
            continue

        offset_zyx = np.array(
            [axis_slice.start for axis_slice in region_box],
            dtype=np.int64,
        )
        global_coordinates = local_coordinates + offset_zyx

        results.append(
            measure_region(
                region_id,
                global_coordinates,
                voxel_size_xyz,
                group_name,
            )
        )

    return results, int(number_of_regions)


def write_csv(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    """Write micropore parameters to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            clean_row = {
                key: (
                    ""
                    if isinstance(value, float) and math.isnan(value)
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(clean_row)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate micropore parameters from a binary XCT TIFF stack."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing sequential binary TIFF slices.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV file.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=(2.4464, 2.4464, 2.4464),
        help=(
            "Physical voxel size in x, y, and z, in micrometres. "
            "Default: 2.4464 2.4464 2.4464."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help=(
            "Voxel values greater than this threshold are treated as pores. "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=6,
        help="Minimum connected-region size retained for analysis. Default: 6.",
    )
    parser.add_argument(
        "--group-name",
        type=str,
        default=None,
        help=(
            "Group name included in the output. "
            "Default: input-folder name."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    group_name = args.group_name or args.input_dir.name

    try:
        print("[1/3] Reading TIFF stack...")
        volume, files = load_tiff_stack(args.input_dir)
        print(
            f"      Loaded {len(files)} slices; volume shape = {volume.shape}."
        )

        print("[2/3] Labeling micropores and calculating parameters...")
        rows, detected_regions = extract_micropore_parameters(
            volume,
            threshold=args.threshold,
            min_voxels=args.min_voxels,
            voxel_size_xyz=args.voxel_size,
            group_name=group_name,
        )
        print(
            f"      Detected {detected_regions} connected regions; "
            f"retained {len(rows)} regions."
        )

        print("[3/3] Writing results...")
        write_csv(rows, args.output_csv)

        metadata = {
            "input_directory": str(args.input_dir),
            "output_csv": str(args.output_csv),
            "group_name": group_name,
            "slice_count": len(files),
            "volume_shape_zyx": list(volume.shape),
            "voxel_size_xyz_um": list(args.voxel_size),
            "threshold": args.threshold,
            "minimum_voxel_count": args.min_voxels,
            "connectivity": 26,
            "detected_region_count": detected_regions,
            "retained_region_count": len(rows),
            "coordinate_convention": (
                "X/Y pixel indices are zero-based; Z slice coordinate is one-based."
            ),
        }
        metadata_path = args.output_csv.with_suffix(".metadata.json")
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

        print(f"Results saved to: {args.output_csv}")
        print(f"Metadata saved to: {metadata_path}")
        return 0

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
