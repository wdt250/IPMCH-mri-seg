from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

from compute_radiomics_agreement_jpg import (
    build_extractor,
    clean_feature_result,
    feature_class,
    label_mask,
    parse_stem,
    prediction_mask,
    summarize_feature_group,
    write_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = ArgumentParser(description="Compute DICOM-space radiomics agreement for manual and automatic ROIs.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--match-csv", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ultralytics-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--include-unmatched", action="store_true")
    return parser.parse_args()


def read_jpg_shape(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image.shape


def read_dicom_2d(path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"Expected a single-slice DICOM, got array shape {array.shape}: {path}")
        array = array[0]
    elif array.ndim != 2:
        raise ValueError(f"Expected a 2D image, got array shape {array.shape}: {path}")

    spacing = image.GetSpacing()
    spacing_2d = (float(spacing[0]), float(spacing[1])) if len(spacing) >= 2 else (1.0, 1.0)
    sitk_image = sitk.GetImageFromArray(array.astype(np.float32))
    sitk_image.SetSpacing(spacing_2d)
    return sitk_image, array.shape, spacing


def mask_to_sitk(mask, spacing):
    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    sitk_mask.SetSpacing((float(spacing[0]), float(spacing[1])))
    return sitk_mask


def extract_features_sitk(extractor, sitk_image, mask, spacing):
    if int(mask.sum()) < 16:
        return {}
    sitk_mask = mask_to_sitk(mask, spacing)
    try:
        return clean_feature_result(extractor.execute(sitk_image, sitk_mask))
    except Exception as exc:
        return {"__error__": str(exc)}


def load_val_matches(match_csv, include_unmatched):
    matches = pd.read_csv(match_csv, dtype=str).fillna("")
    matches = matches[matches["split"] == "val"].copy()
    if not include_unmatched:
        matches = matches[matches["metadata_match"].str.lower() == "yes"].copy()
    return {row.image: row for row in matches.itertuples(index=False)}


def save_recommended(output_dir):
    agreement = pd.read_csv(output_dir / "pyradiomics_feature_agreement.csv")
    stable = agreement[
        (agreement["group"] == "overall")
        & (agreement["ICC2"].astype(float) >= 0.90)
        & (agreement["spearman_rho"].astype(float) >= 0.80)
        & (agreement["spearman_fdr_p"].astype(float) < 0.05)
        & (agreement["median_symmetric_relative_error"].astype(float) <= 0.20)
    ].copy()
    stable = stable.sort_values(["feature_class", "ICC2"], ascending=[True, False])
    stable.to_csv(output_dir / "recommended_stable_features_overall.csv", index=False, encoding="utf-8-sig")
    return stable


def write_readme(output_dir, included, skipped, summary_rows, stable):
    overall = pd.DataFrame(summary_rows)
    overall = overall[overall["group"] == "overall"].copy()
    lines = [
        "# Val DICOM-Space PyRadiomics ICC Summary",
        "",
        f"Included val images: {included}",
        f"Skipped val images: {skipped}",
        "",
        "Auto ROI: DSAM + proto, 150 epochs, `best.pt`.",
        "",
        "Manual ROI: YOLO polygon labels in `labels/val`.",
        "",
        "Image source: original DICOM files. In-plane spacing is read from DICOM by SimpleITK.",
        "",
        "## Overall Feature Class Stability",
        "",
        "| Feature class | Features | ICC >= 0.90 | Median ICC | Stable features |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in overall.sort_values("feature_class").itertuples(index=False):
        lines.append(
            f"| {row.feature_class} | {row.features} | {row.ICC_ge_0_90} | "
            f"{row.median_ICC:.3f} | {row.stable_features} |"
        )
    lines.extend(
        [
            "",
            "Stable feature definition: ICC >= 0.90, Spearman rho >= 0.80, "
            "FDR-adjusted p < 0.05, and median symmetric relative error <= 0.20.",
            "",
            f"Overall stable features: {len(stable)}",
            "",
            "Top stable features by ICC:",
            "",
        ]
    )
    for row in stable.sort_values("ICC2", ascending=False).head(15).itertuples(index=False):
        lines.append(f"- `{row.feature}`: ICC={row.ICC2:.3f}, rho={row.spearman_rho:.3f}")
    (output_dir / "README_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.ultralytics_root.is_dir():
        sys.path.insert(0, str(args.ultralytics_root))

    from ultralytics import YOLO

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = build_extractor(args.output_dir)
    matches = load_val_matches(args.match_csv, args.include_unmatched)

    all_val_images = sorted((args.dataset_root / "images" / "val").glob("*"))
    image_paths = [path for path in all_val_images if path.name in matches]
    skipped_rows = []
    for path in all_val_images:
        if path.name not in matches:
            skipped_rows.append({"image": path.name, "reason": "no_metadata_match"})

    label_dir = args.dataset_root / "labels" / "val"
    model = YOLO(str(args.weights))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        retina_masks=True,
        stream=True,
        verbose=False,
    )

    feature_rows = []
    extraction_errors = []
    sample_rows = []
    for image_path, result in zip(image_paths, results):
        match = matches[image_path.name]
        dicom_path = Path(match.best_dicom_path)
        patient_id, sequence, plane, image_id = parse_stem(image_path.stem)
        try:
            sitk_image, dicom_shape, spacing = read_dicom_2d(dicom_path)
        except Exception as exc:
            skipped_rows.append({"image": image_path.name, "reason": f"dicom_read_error: {exc}"})
            continue

        dicom_h, dicom_w = dicom_shape
        jpg_h, jpg_w = read_jpg_shape(image_path)
        manual_mask = label_mask(label_dir / f"{image_path.stem}.txt", dicom_h, dicom_w)
        auto_mask = prediction_mask(result, jpg_h, jpg_w)
        if auto_mask.shape != (dicom_h, dicom_w):
            auto_mask = cv2.resize(auto_mask, (dicom_w, dicom_h), interpolation=cv2.INTER_NEAREST)

        manual_features = extract_features_sitk(extractor, sitk_image, manual_mask, spacing)
        auto_features = extract_features_sitk(extractor, sitk_image, auto_mask, spacing)

        sample_rows.append(
            {
                "sample_id": image_path.stem,
                "patient_id": patient_id,
                "sequence": sequence,
                "plane": plane,
                "image_id": image_id,
                "dicom_path": str(dicom_path),
                "metadata_match": match.metadata_match,
                "dicom_height": dicom_h,
                "dicom_width": dicom_w,
                "jpg_height": jpg_h,
                "jpg_width": jpg_w,
                "spacing_x": float(spacing[0]) if len(spacing) > 0 else np.nan,
                "spacing_y": float(spacing[1]) if len(spacing) > 1 else np.nan,
                "spacing_z": float(spacing[2]) if len(spacing) > 2 else np.nan,
                "manual_pixels": int(manual_mask.sum()),
                "auto_pixels": int(auto_mask.sum()),
            }
        )
        if "__error__" in manual_features:
            extraction_errors.append({"sample_id": image_path.stem, "method": "manual", "error": manual_features["__error__"]})
            manual_features = {}
        if "__error__" in auto_features:
            extraction_errors.append({"sample_id": image_path.stem, "method": "auto", "error": auto_features["__error__"]})
            auto_features = {}

        for feature in sorted(set(manual_features) | set(auto_features)):
            feature_rows.append(
                {
                    "sample_id": image_path.stem,
                    "patient_id": patient_id,
                    "sequence": sequence,
                    "plane": plane,
                    "image_id": image_id,
                    "feature": feature,
                    "feature_class": feature_class(feature),
                    "manual": manual_features.get(feature, np.nan),
                    "auto": auto_features.get(feature, np.nan),
                }
            )

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(args.output_dir / "pyradiomics_feature_values_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sample_rows).to_csv(args.output_dir / "sample_roi_pixel_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(extraction_errors).to_csv(args.output_dir / "extraction_errors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped_rows).to_csv(args.output_dir / "skipped_images.csv", index=False, encoding="utf-8-sig")

    stat_rows = []
    stat_rows.extend(summarize_feature_group(feature_df, "overall", "all"))
    for plane in ("sag", "cor", "tra"):
        stat_rows.extend(summarize_feature_group(feature_df[feature_df["plane"] == plane], "plane", plane))
    for sequence in ("haste", "trufi"):
        stat_rows.extend(summarize_feature_group(feature_df[feature_df["sequence"] == sequence], "sequence", sequence))

    stat_fieldnames = [
        "group",
        "group_value",
        "feature",
        "feature_class",
        "n_pairs",
        "manual_mean",
        "manual_std",
        "auto_mean",
        "auto_std",
        "ICC2",
        "ICC2_CI95_low",
        "ICC2_CI95_high",
        "ICC_F",
        "ICC_p",
        "spearman_rho",
        "spearman_p",
        "spearman_fdr_p",
        "pearson_r",
        "pearson_p",
        "median_symmetric_relative_error",
        "IQR_symmetric_relative_error",
        "BA_bias",
        "BA_lower_LoA",
        "BA_upper_LoA",
        "proportional_bias_slope",
        "proportional_bias_p",
    ]
    write_csv(args.output_dir / "pyradiomics_feature_agreement.csv", stat_rows, stat_fieldnames)

    summary_rows = []
    stat_df = pd.DataFrame(stat_rows)
    for (group, group_value, cls), subset in stat_df.groupby(["group", "group_value", "feature_class"]):
        icc = subset["ICC2"].astype(float)
        sre = subset["median_symmetric_relative_error"].astype(float)
        stable = subset[
            (subset["ICC2"].astype(float) >= 0.90)
            & (subset["spearman_rho"].astype(float) >= 0.80)
            & (subset["spearman_fdr_p"].astype(float) < 0.05)
            & (subset["median_symmetric_relative_error"].astype(float) <= 0.20)
        ]
        summary_rows.append(
            {
                "group": group,
                "group_value": group_value,
                "feature_class": cls,
                "features": len(subset),
                "ICC_ge_0_90": int((icc >= 0.90).sum()),
                "ICC_ge_0_75": int((icc >= 0.75).sum()),
                "median_ICC": float(np.nanmedian(icc)),
                "mean_ICC": float(np.nanmean(icc)),
                "median_symmetric_relative_error": float(np.nanmedian(sre)),
                "stable_features": len(stable),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "pyradiomics_feature_class_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stable = save_recommended(args.output_dir)
    write_readme(args.output_dir, len(sample_rows), len(skipped_rows), summary_rows, stable)

    print(f"Included {len(sample_rows)} images; skipped {len(skipped_rows)}")
    print(f"Saved {len(feature_rows)} feature rows")
    print(f"Saved {len(stat_rows)} agreement rows")
    print(f"Stable overall features {len(stable)}")
    print(f"Output {args.output_dir}")
    for row in summary_rows:
        if row["group"] == "overall":
            print(
                f"{row['feature_class']}: features={row['features']}, "
                f"median_ICC={row['median_ICC']:.3f}, stable={row['stable_features']}"
            )


if __name__ == "__main__":
    main()
