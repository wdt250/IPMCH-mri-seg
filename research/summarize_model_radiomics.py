"""Summarize radiomic agreement for the three 150-epoch YOLO-based models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "YOLO11": "val_pyradiomics_icc_dicom_260720_yolo11_origin150",
    "DeBiFormerPlus": "val_pyradiomics_icc_dicom_260720_debiformerplus150",
    "DeBiFormerPlus-Consistency": "val_pyradiomics_icc_dicom_260624_proto150",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiomics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plane-dice-long", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260720)
    return parser.parse_args()


def stable_mask(frame: pd.DataFrame, ci85: bool = False) -> pd.Series:
    selected = (
        frame["ICC2"].ge(0.90)
        & frame["spearman_rho"].ge(0.80)
        & frame["spearman_fdr_p"].lt(0.05)
        & frame["median_symmetric_relative_error"].le(0.20)
    )
    if ci85:
        selected &= frame["ICC2_CI95_low"].ge(0.85)
    return selected.fillna(False)


def icc_a1(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 2:
        return np.nan
    n, k = values.shape
    grand = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((column_means - grand) ** 2) / (k - 1)
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else np.nan


def bootstrap_icc_lows(
    values: pd.DataFrame, repetitions: int, seed: int
) -> pd.DataFrame:
    patients = sorted(values["patient_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(patients), size=(repetitions, len(patients)))
    rows = []
    for feature, frame in values.groupby("feature", sort=True):
        arrays = []
        for patient in patients:
            subset = frame[frame["patient_id"].astype(str).eq(patient)]
            arrays.append(subset[["manual", "auto"]].to_numpy(dtype=float))
        estimates = np.empty(repetitions, dtype=float)
        for index, draw in enumerate(draws):
            estimates[index] = icc_a1(np.concatenate([arrays[item] for item in draw]))
        valid = estimates[np.isfinite(estimates)]
        low, median, high = np.percentile(valid, [2.5, 50, 97.5])
        rows.append(
            {
                "feature": feature,
                "participant_bootstrap_CI95_low": low,
                "participant_bootstrap_median": median,
                "participant_bootstrap_CI95_high": high,
                "valid_repetitions": len(valid),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    plane_rows = []
    bootstrap_rows = []
    matched_triplets: set[tuple[str, str]] = set()

    for model_index, (model, directory) in enumerate(MODELS.items()):
        root = args.radiomics_root / directory
        agreement = pd.read_csv(root / "pyradiomics_feature_agreement.csv")
        values = pd.read_csv(root / "pyradiomics_feature_values_long.csv")
        if model == "DeBiFormerPlus-Consistency":
            samples = values.drop_duplicates("sample_id")
            for (patient, sequence), group in samples.groupby(["patient_id", "sequence"]):
                if set(group["plane"].astype(str)) == {"cor", "sag", "tra"}:
                    matched_triplets.add((str(patient), str(sequence)))
        overall = agreement[agreement["group"].eq("overall")].copy()
        overall["stage1"] = stable_mask(overall)
        overall["CI85"] = stable_mask(overall, ci85=True)
        bootstrap = bootstrap_icc_lows(
            values, args.bootstrap_repetitions, args.seed + model_index
        )
        overall = overall.merge(bootstrap, on="feature", how="left")
        overall["cluster_CI85_sensitivity"] = (
            overall["CI85"] & overall["participant_bootstrap_CI95_low"].ge(0.85)
        )
        summary_rows.append(
            {
                "model": model,
                "images": int(values["sample_id"].nunique()),
                "participants": int(values["patient_id"].nunique()),
                "median_ICC": float(overall["ICC2"].median()),
                "stage1_features": int(overall["stage1"].sum()),
                "CI85_candidates": int(overall["CI85"].sum()),
                "CI85_retained_by_participant_bootstrap": int(
                    overall["cluster_CI85_sensitivity"].sum()
                ),
            }
        )
        bootstrap.insert(0, "model", model)
        bootstrap_rows.append(bootstrap)

        plane = agreement[agreement["group"].eq("plane")].copy()
        plane["stage1"] = stable_mask(plane)
        plane["CI85"] = stable_mask(plane, ci85=True)
        for plane_name, frame in plane.groupby("group_value", sort=True):
            plane_rows.append(
                {
                    "model": model,
                    "plane": plane_name,
                    "images": int(frame["n_pairs"].max()),
                    "median_ICC": float(frame["ICC2"].median()),
                    "stage1_features": int(frame["stage1"].sum()),
                    "CI85_candidates": int(frame["CI85"].sum()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    planes = pd.DataFrame(plane_rows)
    variability_rows = []
    for model, frame in planes.groupby("model", sort=False):
        variability_rows.append(
            {
                "model": model,
                "mean_plane_median_ICC": float(frame["median_ICC"].mean()),
                "range_plane_median_ICC": float(
                    frame["median_ICC"].max() - frame["median_ICC"].min()
                ),
                "sd_plane_median_ICC": float(frame["median_ICC"].std(ddof=1)),
                "mean_plane_CI85_candidates": float(frame["CI85_candidates"].mean()),
                "range_plane_CI85_candidates": int(
                    frame["CI85_candidates"].max() - frame["CI85_candidates"].min()
                ),
                "sd_plane_CI85_candidates": float(
                    frame["CI85_candidates"].std(ddof=1)
                ),
            }
        )
    variability = pd.DataFrame(variability_rows)
    if args.plane_dice_long:
        dice_long = pd.read_csv(args.plane_dice_long, dtype={"patient_id": str})
        dice_long = dice_long[
            [
                (str(row.patient_id), str(row.sequence)) in matched_triplets
                for row in dice_long.itertuples(index=False)
            ]
        ].copy()
        model_names = {
            "YOLO11-origin-150ep": "YOLO11",
            "DSAM-baseline-150ep": "DeBiFormerPlus",
            "DSAM-proto-150ep": "DeBiFormerPlus-Consistency",
        }
        dice_long["model"] = dice_long["model"].map(model_names)
        dice_summary = (
            dice_long.groupby("model", as_index=False)["abs_dice_diff"]
            .mean()
            .rename(columns={"abs_dice_diff": "mean_within_triplet_abs_Dice_difference"})
        )
        variability = variability.merge(dice_summary, on="model", how="left")

    summary.to_csv(args.output_dir / "three_model_radiomics_summary.csv", index=False)
    planes.to_csv(args.output_dir / "three_model_plane_agreement.csv", index=False)
    variability.to_csv(args.output_dir / "three_model_plane_variability.csv", index=False)
    pd.concat(bootstrap_rows, ignore_index=True).to_csv(
        args.output_dir / "three_model_participant_bootstrap_feature_icc.csv",
        index=False,
    )

    print("Overall radiomic agreement")
    print(summary.to_string(index=False))
    print("\nPlane-stratified agreement")
    print(planes.to_string(index=False))
    print("\nPlane variability")
    print(variability.to_string(index=False))


if __name__ == "__main__":
    main()
