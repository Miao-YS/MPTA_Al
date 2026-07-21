#!/usr/bin/env python3

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Group",
    "GlobalID",
    "Centroid_X(um)",
    "Centroid_Y(um)",
    "Centroid_Z(um)",
    "EquivalentDiameter(um)",
    "Volume(um^3)",
    "AspectRatio(r3/r1)",
    "Z_thickness(um)",
}


def natural_key(value):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value))]


def number(row, key):
    value = row.get(key, 0.0)
    return float(value) if pd.notna(value) else 0.0


def position(row):
    return np.array([
        number(row, "Centroid_X(um)"),
        number(row, "Centroid_Y(um)"),
        number(row, "Centroid_Z(um)"),
    ])


def distance(a, b):
    return float(np.linalg.norm(position(a) - position(b)))


def mpp(a, b, p):
    d = distance(a, b)
    ml = max(0.0, (p["search_radius"] - d) / p["search_radius"])
    va, vb = number(a, "Volume(um^3)"), number(b, "Volume(um^3)")
    sa, sb = number(a, "AspectRatio(r3/r1)"), number(b, "AspectRatio(r3/r1)")
    mv = min(va, vb) / max(va, vb) if max(va, vb) > 0 else 0.0
    ms = min(sa, sb) / max(sa, sb) if max(sa, sb) > 0 else 0.0
    return p["w_distance"] * ml + p["w_volume"] * mv + p["w_shape"] * ms


def weighted_centroid(rows):
    volumes = np.array([number(row, "Volume(um^3)") for row in rows], dtype=float)
    points = np.array([position(row) for row in rows], dtype=float)
    total = volumes.sum()
    return np.average(points, axis=0, weights=volumes) if total > 0 else points.mean(axis=0)


def local_translation(target, matches, frame_a, frame_b, limit):
    target_pos = position(target)
    refs = []
    for item in matches:
        if item["Frame1"] != frame_a or item["Frame2"] != frame_b:
            continue
        source_pos = np.array([item["Hole1_X"], item["Hole1_Y"], item["Hole1_Z"]], dtype=float)
        target_ref = np.array([item["Hole2_X"], item["Hole2_Y"], item["Hole2_Z"]], dtype=float)
        refs.append((np.linalg.norm(target_ref - target_pos), target_ref - source_pos))
    if not refs:
        return np.zeros(3)
    refs.sort(key=lambda x: x[0])
    return np.median(np.array([x[1] for x in refs[:limit]]), axis=0)


def load_tables(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def one_to_one(previous, current, frame_a, frame_b, p):
    candidates = []
    for a in previous:
        scores = sorted(
            ((mpp(a, b, p), b) for b in current),
            key=lambda x: x[0],
            reverse=True,
        )
        if not scores:
            continue
        best, target = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        if best >= p["mpp_threshold"] and best - second >= p["mpp_gap"]:
            candidates.append((best, second, a, target))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_a, used_b, rows = set(), set(), []

    for best, second, a, b in candidates:
        aid, bid = str(a["GlobalID"]), str(b["GlobalID"])
        if aid in used_a or bid in used_b:
            continue
        used_a.add(aid)
        used_b.add(bid)
        rows.append({
            "Frame1": frame_a,
            "Hole1_ID": aid,
            "Hole1_X": number(a, "Centroid_X(um)"),
            "Hole1_Y": number(a, "Centroid_Y(um)"),
            "Hole1_Z": number(a, "Centroid_Z(um)"),
            "Frame2": frame_b,
            "Hole2_ID": bid,
            "Hole2_X": number(b, "Centroid_X(um)"),
            "Hole2_Y": number(b, "Centroid_Y(um)"),
            "Hole2_Z": number(b, "Centroid_Z(um)"),
            "Best_MPP": best,
            "Second_MPP": second,
            "RelationType": "match",
        })

    return rows, used_a, used_b


def coalescence(previous, current, frame_a, frame_b, matches, used_targets, p):
    rows = []
    used_sources = set()

    for target in current:
        tid = str(target["GlobalID"])
        if tid in used_targets:
            continue

        candidates = sorted(previous, key=lambda row: distance(row, target))[:p["candidate_count"]]
        best = None

        for size in range(2, min(len(candidates), p["max_combo"]) + 1):
            for combo in itertools.combinations(candidates, size):
                target_volume = number(target, "Volume(um^3)")
                source_volume = sum(number(row, "Volume(um^3)") for row in combo)
                if target_volume <= 0:
                    continue

                volume_error = abs(source_volume - target_volume) / target_volume
                if volume_error > p["volume_tolerance"]:
                    continue

                centroid = weighted_centroid(combo)
                corrected = centroid + local_translation(
                    target,
                    matches,
                    frame_a,
                    frame_b,
                    p["candidate_count"],
                )
                residual = float(np.linalg.norm(corrected - position(target)))

                if residual > p["centroid_tolerance"]:
                    continue

                rank = (volume_error, residual, size)
                if best is None or rank < best[0]:
                    best = (rank, combo, source_volume, corrected, residual)

        if best is None:
            continue

        _, combo, source_volume, corrected, residual = best
        source_ids = [str(row["GlobalID"]) for row in combo]
        used_sources.update(source_ids)
        used_targets.add(tid)

        rows.append({
            "Frame1": frame_a,
            "Source_HoleIDs": ";".join(source_ids),
            "Source_Count": len(source_ids),
            "Source_TotalVolume": source_volume,
            "Corrected_Centroid_X": corrected[0],
            "Corrected_Centroid_Y": corrected[1],
            "Corrected_Centroid_Z": corrected[2],
            "Frame2": frame_b,
            "Hole2_ID": tid,
            "Hole2_Volume": number(target, "Volume(um^3)"),
            "Residual_Distance": residual,
            "RelationType": "merge",
        })

    return rows, used_sources, used_targets


def nucleation(previous, current, next_frame, frame_name, used_targets, p):
    rows = []

    if next_frame is None:
        return rows

    for target in current:
        tid = str(target["GlobalID"])
        if tid in used_targets:
            continue
        if number(target, "Z_thickness(um)") <= p["minimum_thickness"]:
            continue
        if number(target, "EquivalentDiameter(um)") < p["minimum_diameter"]:
            continue

        previous_max = max((mpp(row, target, p) for row in previous), default=0.0)
        if previous_max >= p["nucleation_previous_mpp"]:
            continue

        scores = sorted(
            ((mpp(target, row, p), row) for row in next_frame),
            key=lambda x: x[0],
            reverse=True,
        )
        if not scores or scores[0][0] < p["nucleation_confirmation_mpp"]:
            continue

        score, confirmed = scores[0]
        rows.append({
            "Frame": frame_name,
            "Hole_ID": tid,
            "Previous_Max_MPP": previous_max,
            "Next_Hole_ID": str(confirmed["GlobalID"]),
            "Next_MPP": score,
            "RelationType": "nucleation",
        })

    return rows


def run(df, output_dir, p):
    groups = sorted(df["Group"].astype(str).unique(), key=natural_key)
    all_matches, all_merges, all_nucleations = [], [], []

    for index, (frame_a, frame_b) in enumerate(zip(groups[:-1], groups[1:])):
        previous = df[df["Group"].astype(str) == frame_a].to_dict("records")
        current = df[df["Group"].astype(str) == frame_b].to_dict("records")
        following = None
        if index + 2 < len(groups):
            next_name = groups[index + 2]
            following = df[df["Group"].astype(str) == next_name].to_dict("records")

        matches, used_sources, used_targets = one_to_one(
            previous,
            current,
            frame_a,
            frame_b,
            p,
        )

        merges, merged_sources, used_targets = coalescence(
            previous,
            current,
            frame_a,
            frame_b,
            matches,
            used_targets,
            p,
        )

        if merged_sources:
            matches = [
                row for row in matches
                if row["Hole1_ID"] not in merged_sources
            ]

        nuclei = nucleation(
            previous,
            current,
            following,
            frame_b,
            used_targets,
            p,
        )

        all_matches.extend(matches)
        all_merges.extend(merges)
        all_nucleations.extend(nuclei)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_matches).to_csv(output_dir / "match.csv", index=False)
    pd.DataFrame(all_merges).to_csv(output_dir / "merge.csv", index=False)
    pd.DataFrame(all_nucleations).to_csv(output_dir / "nucleation.csv", index=False)

    with (output_dir / "run_parameters.json").open("w", encoding="utf-8") as file:
        json.dump(p, file, indent=2)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--search-radius", required=True, type=float)
    parser.add_argument("--w-distance", required=True, type=float)
    parser.add_argument("--w-volume", required=True, type=float)
    parser.add_argument("--w-shape", required=True, type=float)
    parser.add_argument("--mpp-threshold", required=True, type=float)
    parser.add_argument("--mpp-gap", required=True, type=float)
    parser.add_argument("--volume-tolerance", required=True, type=float)
    parser.add_argument("--centroid-tolerance", required=True, type=float)
    parser.add_argument("--candidate-count", required=True, type=int)
    parser.add_argument("--max-combo", required=True, type=int)
    parser.add_argument("--minimum-thickness", required=True, type=float)
    parser.add_argument("--minimum-diameter", required=True, type=float)
    parser.add_argument("--nucleation-previous-mpp", required=True, type=float)
    parser.add_argument("--nucleation-confirmation-mpp", required=True, type=float)
    return parser.parse_args()


def main():
    args = arguments()
    p = vars(args).copy()
    csv_paths = p.pop("csv")
    output_dir = p.pop("output_dir")

    if not np.isclose(p["w_distance"] + p["w_volume"] + p["w_shape"], 1.0):
        raise ValueError("MPP weights must sum to 1.")

    df = load_tables(csv_paths)
    run(df, output_dir, p)


if __name__ == "__main__":
    main()
