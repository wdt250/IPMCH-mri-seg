from argparse import ArgumentParser
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pingouin as pg
import SimpleITK as sitk

from compute_radiomics_agreement_jpg import label_mask, parse_stem, prediction_mask


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = ArgumentParser(description="Run participant-level ICC sensitivity and optional ROI quality control.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ultralytics-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--qc-sample", help="Optional de-identified image stem for an ROI overlay.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    return parser.parse_args()


def icc_a1(feature_df):
    feature_df = feature_df.dropna(subset=["manual", "auto"]).copy()
    if len(feature_df) < 3:
        return np.nan, np.nan, np.nan
    manual = feature_df["manual"].to_numpy(dtype=float)
    auto = feature_df["auto"].to_numpy(dtype=float)
    if not np.isfinite(manual).all() or not np.isfinite(auto).all():
        return np.nan, np.nan, np.nan
    long_rows = []
    for row in feature_df.itertuples(index=False):
        long_rows.append({"sample_id": row.sample_id, "method": "manual", "value": float(row.manual)})
        long_rows.append({"sample_id": row.sample_id, "method": "auto", "value": float(row.auto)})
    long_df = pd.DataFrame(long_rows)
    try:
        result = pg.intraclass_corr(
            data=long_df,
            targets="sample_id",
            raters="method",
            ratings="value",
            nan_policy="omit",
        )
        row = result[result["Type"].isin(["ICC2", "ICC(A,1)"])].iloc[0]
        ci_column = "CI95%" if "CI95%" in result.columns else "CI95"
        ci = row[ci_column]
        return float(row["ICC"]), float(ci[0]), float(ci[1])
    except Exception:
        return np.nan, np.nan, np.nan


def leave_one_patient_out(result_dir):
    values = pd.read_csv(result_dir / "pyradiomics_feature_values_long.csv")
    agreement = pd.read_csv(result_dir / "pyradiomics_feature_agreement.csv")
    full = agreement[agreement["group"].eq("overall")].copy()
    full = full.set_index("feature")
    patients = sorted(values["patient_id"].astype(str).unique())

    rows = []
    for feature, feature_df in values.groupby("feature"):
        full_icc = float(full.loc[feature, "ICC2"]) if feature in full.index else np.nan
        feature_class = str(feature_df["feature_class"].iloc[0])
        for patient in patients:
            subset = feature_df[feature_df["patient_id"].astype(str) != patient].copy()
            lopo_icc, ci_low, ci_high = icc_a1(subset)
            rows.append(
                {
                    "feature": feature,
                    "feature_class": feature_class,
                    "leave_patient": patient,
                    "n_pairs": int(subset.dropna(subset=["manual", "auto"]).shape[0]),
                    "full_ICC2": full_icc,
                    "lopo_ICC2": lopo_icc,
                    "lopo_CI95_low": ci_low,
                    "lopo_CI95_high": ci_high,
                    "ICC_drop": full_icc - lopo_icc if np.isfinite(full_icc) and np.isfinite(lopo_icc) else np.nan,
                    "drop_ge_0_10": bool((full_icc - lopo_icc) >= 0.10) if np.isfinite(full_icc) and np.isfinite(lopo_icc) else False,
                }
            )
    lopo = pd.DataFrame(rows)
    lopo.to_csv(result_dir / "leave_one_patient_out_icc.csv", index=False, encoding="utf-8-sig")

    feature_summary_rows = []
    for feature, subset in lopo.groupby("feature"):
        subset = subset.copy()
        if subset["ICC_drop"].notna().any():
            max_row = subset.loc[subset["ICC_drop"].idxmax()]
            min_lopo = float(subset["lopo_ICC2"].min())
            max_drop = float(max_row["ICC_drop"])
            max_drop_patient = str(max_row["leave_patient"])
        else:
            min_lopo = np.nan
            max_drop = np.nan
            max_drop_patient = ""
        feature_summary_rows.append(
            {
                "feature": feature,
                "feature_class": str(subset["feature_class"].iloc[0]),
                "full_ICC2": float(subset["full_ICC2"].iloc[0]),
                "min_lopo_ICC2": min_lopo,
                "max_ICC_drop": max_drop,
                "max_drop_patient": max_drop_patient,
                "n_patients_drop_ge_0_10": int(subset["drop_ge_0_10"].sum()),
            }
        )
    feature_summary = pd.DataFrame(feature_summary_rows)
    feature_summary.to_csv(result_dir / "leave_one_patient_out_feature_summary.csv", index=False, encoding="utf-8-sig")

    patient_class_rows = []
    for (patient, feature_class), subset in lopo.groupby(["leave_patient", "feature_class"]):
        drops = subset["ICC_drop"].astype(float)
        patient_class_rows.append(
            {
                "leave_patient": patient,
                "feature_class": feature_class,
                "features": len(subset),
                "median_ICC_drop": float(np.nanmedian(drops)),
                "max_ICC_drop": float(np.nanmax(drops)),
                "features_drop_ge_0_10": int((drops >= 0.10).sum()),
            }
        )
    patient_class_summary = pd.DataFrame(patient_class_rows)
    patient_class_summary.to_csv(
        result_dir / "leave_one_patient_out_patient_class_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return lopo, feature_summary, patient_class_summary


def filter_stable_features(result_dir):
    agreement = pd.read_csv(result_dir / "pyradiomics_feature_agreement.csv")
    base = agreement[
        (agreement["group"] == "overall")
        & (agreement["ICC2"].astype(float) >= 0.90)
        & (agreement["ICC2_CI95_low"].astype(float) >= 0.85)
        & (agreement["spearman_rho"].astype(float) >= 0.80)
        & (agreement["spearman_fdr_p"].astype(float) < 0.05)
        & (agreement["median_symmetric_relative_error"].astype(float) <= 0.20)
    ].copy()
    base = base.sort_values(["feature_class", "ICC2"], ascending=[True, False])
    base.to_csv(result_dir / "recommended_stable_features_overall_ci85.csv", index=False, encoding="utf-8-sig")

    by_sequence = agreement[
        (agreement["group"] == "sequence")
        & (agreement["ICC2"].astype(float) >= 0.90)
        & (agreement["ICC2_CI95_low"].astype(float) >= 0.85)
        & (agreement["spearman_rho"].astype(float) >= 0.80)
        & (agreement["spearman_fdr_p"].astype(float) < 0.05)
        & (agreement["median_symmetric_relative_error"].astype(float) <= 0.20)
    ].copy()
    by_sequence = by_sequence.sort_values(["group_value", "feature_class", "ICC2"], ascending=[True, True, False])
    by_sequence.to_csv(result_dir / "recommended_stable_features_by_sequence_ci85.csv", index=False, encoding="utf-8-sig")
    return base, by_sequence


def read_image_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def normalize_dicom_for_display(dicom_path):
    image = sitk.ReadImage(str(dicom_path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 3:
        array = array[0]
    array = array.astype(np.float32)
    lo, hi = np.percentile(array, [1, 99])
    if hi <= lo:
        lo, hi = float(np.min(array)), float(np.max(array))
    disp = np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1)
    return (disp * 255).astype(np.uint8)


def save_png_unicode(path, image):
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Failed to encode png: {path}")
    Path(path).write_bytes(buf.tobytes())


def draw_mask_overlay(base, manual_mask, auto_mask):
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = rgb.copy()
    manual_only = (manual_mask == 1) & (auto_mask == 0)
    auto_only = (auto_mask == 1) & (manual_mask == 0)
    overlap = (manual_mask == 1) & (auto_mask == 1)
    overlay[manual_only] = (0, 210, 0)
    overlay[auto_only] = (0, 0, 255)
    overlay[overlap] = (0, 210, 255)
    rgb = cv2.addWeighted(overlay, 0.35, rgb, 0.65, 0)
    for mask, color in ((manual_mask, (0, 255, 0)), (auto_mask, (0, 0, 255))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, color, 1)
    return rgb


def make_qc_overlay(args):
    if args.ultralytics_root.is_dir():
        sys.path.insert(0, str(args.ultralytics_root))
    from ultralytics import YOLO

    sample = args.qc_sample
    image_path = args.dataset_root / "images" / "val" / f"{sample}.jpg"
    label_path = args.dataset_root / "labels" / "val" / f"{sample}.txt"
    sample_rows = pd.read_csv(args.result_dir / "sample_roi_pixel_counts.csv")
    row = sample_rows[sample_rows["sample_id"].eq(sample)].iloc[0]
    dicom_path = Path(row["dicom_path"])

    dicom_display = normalize_dicom_for_display(dicom_path)
    dicom_h, dicom_w = dicom_display.shape
    jpg = read_image_unicode(image_path)
    jpg_h, jpg_w = jpg.shape

    model = YOLO(str(args.weights))
    result = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        retina_masks=True,
        verbose=False,
    )[0]
    manual_mask = label_mask(label_path, dicom_h, dicom_w)
    auto_mask = prediction_mask(result, jpg_h, jpg_w)
    if auto_mask.shape != (dicom_h, dicom_w):
        auto_mask = cv2.resize(auto_mask, (dicom_w, dicom_h), interpolation=cv2.INTER_NEAREST)

    overlay = draw_mask_overlay(dicom_display, manual_mask, auto_mask)
    output = args.result_dir / f"qc_overlay_{sample}.png"
    save_png_unicode(output, overlay)
    return {
        "sample_id": sample,
        "output": str(output),
        "manual_pixels": int(manual_mask.sum()),
        "auto_pixels": int(auto_mask.sum()),
        "auto_manual_ratio": float(auto_mask.sum() / max(manual_mask.sum(), 1)),
    }


def write_summary(result_dir, stable_ci85, by_sequence_ci85, feature_summary, patient_class_summary, qc):
    strong_drops = feature_summary[feature_summary["max_ICC_drop"].astype(float) >= 0.10].copy()
    patient_flags = patient_class_summary[patient_class_summary["features_drop_ge_0_10"].astype(int) > 0].copy()
    lines = [
        "# DICOM ICC Sensitivity and QC",
        "",
        "ICC here measures segmentation agreement between manual ROI and automatic ROI. It is not test-retest repeatability.",
        "",
        "Participant-level clustering was addressed with leave-one-participant-out analysis.",
        "",
        f"Overall stable features after CI lower-bound >= 0.85 filter: {len(stable_ci85)}",
        "",
        "Stable features by sequence after the same filter:",
        "",
    ]
    for seq, subset in by_sequence_ci85.groupby("group_value"):
        lines.append(f"- {seq}: {len(subset)} features")
    lines.extend(
        [
            "",
            f"Features with leave-one-patient-out ICC drop >= 0.10: {len(strong_drops)}",
            "",
        ]
    )
    if not strong_drops.empty:
        for row in strong_drops.sort_values("max_ICC_drop", ascending=False).head(20).itertuples(index=False):
            lines.append(
                f"- `{row.feature}` ({row.feature_class}): max drop={row.max_ICC_drop:.3f}, "
                f"leave patient={row.max_drop_patient}"
            )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "Patient/class groups with at least one feature dropping >= 0.10:",
            "",
        ]
    )
    if not patient_flags.empty:
        for row in patient_flags.sort_values(["leave_patient", "feature_class"]).itertuples(index=False):
            lines.append(
                f"- patient {row.leave_patient}, {row.feature_class}: "
                f"{row.features_drop_ge_0_10}/{row.features} features, max drop={row.max_ICC_drop:.3f}"
            )
    else:
        lines.append("- None")
    if qc is not None:
        lines.extend(
            [
                "",
                "QC overlay:",
                "",
                f"- sample `{qc['sample_id']}`: manual={qc['manual_pixels']}, auto={qc['auto_pixels']}, "
                f"auto/manual ratio={qc['auto_manual_ratio']:.2f}",
                f"- overlay: `{Path(qc['output']).name}`",
                "",
                "Color legend: green=manual-only, red=auto-only, yellow=overlap.",
            ]
        )
    (result_dir / "README_sensitivity_qc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    lopo, feature_summary, patient_class_summary = leave_one_patient_out(args.result_dir)
    stable_ci85, by_sequence_ci85 = filter_stable_features(args.result_dir)
    qc = make_qc_overlay(args) if args.qc_sample else None
    write_summary(args.result_dir, stable_ci85, by_sequence_ci85, feature_summary, patient_class_summary, qc)

    print(f"LOPO rows: {len(lopo)}")
    print(f"Stable CI>=0.85 overall features: {len(stable_ci85)}")
    print("Stable CI>=0.85 by sequence:")
    print(by_sequence_ci85.groupby("group_value").size().to_string())
    if qc is not None:
        print(f"QC overlay: {qc['output']}")
        print(f"QC ratio auto/manual: {qc['auto_manual_ratio']:.3f}")


if __name__ == "__main__":
    main()
