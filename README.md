# MPTA_AI

Code and example data for the analysis of three-dimensional micropore evolution in cast Al–Si alloys using in situ X-ray computed tomography (XCT).

## Overview

This repository provides a simplified and functionally equivalent implementation of the main computational workflow used to characterize and track micropores across successive tensile-loading stages.

The workflow contains three modules:

1. **Spatial registration** of sequential XCT TIFF stacks.
2. **Micropore parameter calculation** from segmented binary TIFF stacks.
3. **Micropore tracking** across loading stages, including:
   - one-to-one micropore matching;
   - micropore coalescence identification; and
   - micropore nucleation detection.

The repository is intended to improve the transparency and reproducibility of the image-analysis workflow. It does not include XCT reconstruction, denoising, or grayscale segmentation. Those preprocessing steps must be completed before the binary TIFF stacks are supplied to the parameter-calculation script.

## Repository contents

| File | Purpose |
|---|---|
| `spatial_registration.py` | Performs translation-only registration between a reference XCT TIFF stack and a moving XCT TIFF stack. |
| `micropore_parameter_calculation.py` | Reconstructs a binary TIFF stack as a 3D volume, labels micropores using 26-neighbour connectivity, and calculates micropore-level geometric parameters. |
| `micropore_tracking.py` | Tracks micropores across successive loading stages using the Matching Probability Parameter (MPP), coalescence criteria, and nucleation criteria. |
| `data.xlsx` | Example data for successful one-to-one micropore tracking|
| `requirements.txt` | Python package requirements. |
| `LICENSE` | MIT open-source license. |

## Computational workflow

```text
Reconstructed XCT TIFF stacks
            |
            v
Spatial registration
            |
            v
Image segmentation performed externally
            |
            v
Binary TIFF stacks
            |
            v
Micropore parameter calculation
            |
            v
Micropore-level CSV tables
            |
            v
One-to-one matching
            |
            v
Coalescence identification
            |
            v
Nucleation detection
            |
            v
Tracking-result CSV files
```

## Software requirements

Python **3.10 or newer** is required.

The scripts use the following packages:

- NumPy
- pandas
- SciPy
- Pillow
- SimpleITK

Install the dependencies with:

```bash
pip install -r requirements.txt
```

A virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Suggested directory structure

```text
MPTA_AI/
├── README.md
├── LICENSE
├── requirements.txt
├── spatial_registration.py
├── micropore_parameter_calculation.py
├── micropore_tracking.py
├── data/
│   ├── reference/
│   ├── moving/
│   ├── binary_stage_1/
│   ├── binary_stage_2/
│   └── binary_stage_3/
└── results/
    ├── registered/
    ├── parameters/
    └── tracking/
```

TIFF files within each stack should have consistent image dimensions and filenames that preserve the correct slice order.

---

# 1. Spatial registration

## Purpose

`spatial_registration.py` reads a reference TIFF stack and a moving TIFF stack, reconstructs them as 3D volumes, and performs translation-only registration using Mattes mutual information and gradient-descent optimization.

The moving image is resampled on its original grid so that its original dimensions, spacing, origin, and direction are preserved.

## Input

- A folder containing the reference `.tif` or `.tiff` slices.
- A folder containing the moving `.tif` or `.tiff` slices.
- Voxel spacing in the x, y, and z directions.

All slices in a stack must have consistent dimensions.

## Example

```bash
python spatial_registration.py \
  --reference-dir data/reference \
  --moving-dir data/moving \
  --output-dir results/registered \
  --spacing 2.4464 2.4464 2.4464
```

For Windows Command Prompt, the command can be entered on one line:

```bash
python spatial_registration.py --reference-dir data/reference --moving-dir data/moving --output-dir results/registered --spacing 2.4464 2.4464 2.4464
```

Optional registration settings can also be supplied:

```text
--shrink-factor
--histogram-bins
--sampling-percentage
--seed
--learning-rate
--iterations
```

Display all available options with:

```bash
python spatial_registration.py --help
```

## Output

The output folder contains:

- registered TIFF slices;
- `translation_transform.tfm`, containing the estimated transformation; and
- `registration_summary.json`, containing the input dimensions, voxel spacing, optimizer settings, random seed, final translation, final metric value, and stopping condition.

The fixed random seed used for metric sampling makes the stochastic sampling step reproducible when the same software environment and input data are used.

---

# 2. Micropore parameter calculation

## Purpose

`micropore_parameter_calculation.py` reads a segmented binary XCT TIFF stack and reconstructs it as a 3D array. Voxels with intensity values greater than the selected threshold are treated as micropore voxels.

Individual micropores are identified using **26-neighbour three-dimensional connectivity**. Connected regions smaller than the selected minimum voxel count are excluded.

## Input

- A folder containing sequential binary `.tif` or `.tiff` slices.
- Physical voxel size in x, y, and z.
- Binary threshold.
- Minimum connected-region size in voxels.
- Optional loading-stage or group name.

The script assumes that image reconstruction, denoising, and segmentation have already been completed.

## Example

```bash
python micropore_parameter_calculation.py \
  --input-dir data/binary_stage_1 \
  --output-csv results/parameters/stage_1_parameters.csv \
  --voxel-size 2.4464 2.4464 2.4464 \
  --threshold 0 \
  --min-voxels 6 \
  --group-name S1
```

For Windows Command Prompt:

```bash
python micropore_parameter_calculation.py --input-dir data/binary_stage_1 --output-csv results/parameters/stage_1_parameters.csv --voxel-size 2.4464 2.4464 2.4464 --threshold 0 --min-voxels 6 --group-name S1
```

Display all available options with:

```bash
python micropore_parameter_calculation.py --help
```

## Calculated parameters

The output CSV contains:

- global and regional micropore identifiers;
- centroid coordinates in voxel indices;
- centroid coordinates in physical units;
- voxel count;
- physical volume;
- equivalent diameter;
- occupied-slice count;
- z-direction thickness;
- short-to-long principal-axis ratio; and
- normalized principal variances, `Lambda1` and `Lambda2`.

## Coordinate convention

- X and Y pixel indices are zero-based.
- Z is reported using one-based slice numbering.
- Physical coordinates are calculated using the supplied voxel spacing.

## Output

For an output path such as:

```text
results/parameters/stage_1_parameters.csv
```

the script generates:

```text
stage_1_parameters.csv
stage_1_parameters.metadata.json
```

The metadata file records the input path, volume shape, voxel size, threshold, minimum voxel count, connectivity, and number of detected and retained regions.

---

# 3. Micropore tracking

## Purpose

`micropore_tracking.py` reads micropore-level CSV tables from successive loading stages and applies the MPTA decision sequence:

1. one-to-one matching;
2. coalescence identification; and
3. nucleation detection.

The algorithm compares spatial position, volume, and shape using the Matching Probability Parameter:

```text
MPP = w_distance × M_L + w_volume × M_V + w_shape × M_S
```

The three weights must sum to 1.

### One-to-one matching

For each candidate micropore pair, the script calculates the MPP. A correspondence is accepted when:

- the highest MPP reaches the specified minimum threshold; and
- the difference between the highest and second-highest MPP reaches the specified gap threshold.

Each source micropore and each target micropore can participate in only one accepted one-to-one match.

### Coalescence identification

For unmatched target micropores, the script:

- selects the nearest source-micropore candidates;
- enumerates source combinations;
- compares the summed source volume with the target volume;
- calculates the volume-weighted source centroid;
- estimates local translation from neighbouring one-to-one matches;
- corrects the source-group centroid; and
- compares the residual centroid distance with a prescribed tolerance.

The public implementation uses a user-supplied residual-distance tolerance. It does not automatically calculate an adaptive threshold.

### Nucleation detection

An unmatched and non-coalesced micropore is considered a nucleation candidate when:

- its maximum MPP with micropores in the previous loading stage is below the specified threshold;
- it exceeds the minimum thickness and equivalent-diameter criteria; and
- it is detected again in the immediately following loading stage with an MPP reaching the confirmation threshold.

A micropore in the final loading stage cannot be confirmed as a nucleation event because no subsequent stage is available.

## Required CSV columns

Each input CSV must contain:

```text
Group
GlobalID
Centroid_X(um)
Centroid_Y(um)
Centroid_Z(um)
EquivalentDiameter(um)
Volume(um^3)
AspectRatio(r3/r1)
Z_thickness(um)
```

The `Group` field identifies the loading stage. Group names should preserve the intended sequence, for example:

```text
S1, S2, S3, S4, S5
```

## Important column-name compatibility note

The current parameter-calculation script writes several column names using underscore notation, whereas the tracking script expects parenthesized unit notation.

Before running the tracking script, rename the following columns:

| Parameter-calculation output | Tracking input |
|---|---|
| `Centroid_X_um` | `Centroid_X(um)` |
| `Centroid_Y_um` | `Centroid_Y(um)` |
| `Centroid_Z_um` | `Centroid_Z(um)` |
| `Volume_um3` | `Volume(um^3)` |
| `EquivalentDiameter_um` | `EquivalentDiameter(um)` |
| `AspectRatio_short_long` | `AspectRatio(r3/r1)` |
| `Z_thickness_um` | `Z_thickness(um)` |

A simple conversion can be performed with pandas:

```python
import pandas as pd

df = pd.read_csv("stage_1_parameters.csv")

df = df.rename(columns={
    "Centroid_X_um": "Centroid_X(um)",
    "Centroid_Y_um": "Centroid_Y(um)",
    "Centroid_Z_um": "Centroid_Z(um)",
    "Volume_um3": "Volume(um^3)",
    "EquivalentDiameter_um": "EquivalentDiameter(um)",
    "AspectRatio_short_long": "AspectRatio(r3/r1)",
    "Z_thickness_um": "Z_thickness(um)",
})

df.to_csv("stage_1_tracking_input.csv", index=False)
```

For a fully automated public workflow, the column names in the two scripts should eventually be unified.

## Input-file format

`micropore_tracking.py` reads **CSV files**, not Excel workbooks. If `data.xlsx` contains the analysis data, export each loading stage to CSV before running the tracking script.

Example:

```text
data/stage_1.csv
data/stage_2.csv
data/stage_3.csv
```

## Example command

```bash
python micropore_tracking.py \
  data/stage_1.csv \
  data/stage_2.csv \
  data/stage_3.csv \
  --output-dir results/tracking \
  --search-radius 200 \
  --w-distance 0.8 \
  --w-volume 0.1 \
  --w-shape 0.1 \
  --mpp-threshold 0.8 \
  --mpp-gap 0.1 \
  --volume-tolerance 0.1 \
  --centroid-tolerance 40 \
  --candidate-count 9 \
  --max-combo 9 \
  --minimum-thickness 2.6608 \
  --minimum-diameter 12 \
  --nucleation-previous-mpp 0.8 \
  --nucleation-confirmation-mpp 0.8
```

For Windows Command Prompt, enter the command on one line.

Display all available options with:

```bash
python micropore_tracking.py --help
```

## Tracking output

The tracking output folder contains:

| File | Description |
|---|---|
| `match.csv` | Accepted one-to-one micropore correspondences. |
| `merge.csv` | Identified many-to-one micropore coalescence events. |
| `nucleation.csv` | Confirmed micropore nucleation events. |
| `run_parameters.json` | Complete set of parameter values used for the run. |

---

# Input and output summary

| Module | Input | Output |
|---|---|---|
| Spatial registration | Reference and moving XCT TIFF stacks | Registered TIFF stack, transform file, registration summary |
| Parameter calculation | Segmented binary TIFF stack | Micropore-level CSV and metadata JSON |
| Micropore tracking | Micropore-level CSV tables from successive stages | One-to-one, coalescence, and nucleation CSV tables |

## Reproducibility notes

To reproduce a published analysis, report and preserve:

- exact Python and package versions;
- voxel spacing;
- segmentation procedure and threshold;
- minimum voxel count;
- registration settings and random seed;
- MPP weights;
- MPP acceptance and gap thresholds;
- candidate-neighbour count;
- coalescence volume tolerance;
- residual centroid-distance tolerance;
- minimum micropore thickness;
- minimum equivalent diameter; and
- nucleation thresholds.

The same physical voxel size must be used consistently in registration, micropore parameter calculation, manuscript reporting, and noise-removal criteria.

## Limitations

- XCT reconstruction, denoising, and segmentation are not included.
- The scripts provide a simplified equivalent workflow rather than the complete internal research code.
- Coalescence identification uses a prescribed residual centroid-distance tolerance in the public implementation.
- Tracking results depend on image resolution, segmentation quality, loading-stage interval, and selected parameters.
- The full publication figures require separate statistical and plotting scripts.

## Citation

Please cite the associated article when using this repository. A permanent repository DOI can be added here after archiving the release through Zenodo.

## License

This project is released under the MIT License.
