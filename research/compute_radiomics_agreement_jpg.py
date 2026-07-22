from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pingouin as pg


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = ArgumentParser(description="Compute radiomics agreement in JPG pixel space.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ultralytics-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    return parser.parse_args()


def parse_stem(stem):
    left, image_id = stem.rsplit("_", 1)
    patient_id, sequence_plane = left.split("_", 1)
    sequence, plane = sequence_plane.rsplit("-", 1)
    return patient_id, sequence, plane, image_id


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image.astype(np.float32)


def label_mask(label_path, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] = np.clip(coords[:, 0] * width, 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1] * height, 0, height - 1)
        cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
    return mask


def prediction_mask(result, height, width):
    if result.masks is None or len(result.masks.data) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    masks = result.masks.data.detach().cpu().numpy()
    union = np.any(masks > 0.5, axis=0).astype(np.uint8)
    if union.shape != (height, width):
        union = cv2.resize(union, (width, height), interpolation=cv2.INTER_NEAREST)
    return union


def to_sitk(image, mask):
    # This analysis uses JPG-derived 2D images, so spacing is pixel units.
    sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    sitk_image.SetSpacing((1.0, 1.0))
    sitk_mask.SetSpacing((1.0, 1.0))
    return sitk_image, sitk_mask


def build_extractor(output_dir):
    params = {
        "setting": {
            "label": 1,
            "normalize": True,
            "normalizeScale": 100,
            "binWidth": 10,
            "interpolator": "sitkBSpline",
            "correctMask": True,
            "minimumROIDimensions": 2,
            "minimumROISize": 16,
            "geometryTolerance": 1e-6,
        },
        "imageType": {"Original": {}},
        "featureClass": {
            "shape2D": [],
            "firstorder": [],
            "glcm": [],
            "glrlm": [],
            "glszm": [],
            "gldm": [],
            "ngtdm": [],
        },
    }
    # Avoid requiring PyYAML; save JSON with the same parameter structure.
    (output_dir / "pyradiomics_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    extractor = featureextractor.RadiomicsFeatureExtractor(**params["setting"])
    extractor.disableAllFeatures()
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    for feature_class in params["featureClass"]:
        extractor.enableFeatureClassByName(feature_class)
    return extractor


def clean_feature_result(result):
    cleaned = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        if not key.startswith("original_"):
            continue
        try:
            cleaned[key] = float(value)
        except Exception:
            continue
    return cleaned


def extract_features(extractor, image, mask):
    if int(mask.sum()) < 16:
        return {}
    sitk_image, sitk_mask = to_sitk(image, mask)
    try:
        return clean_feature_result(extractor.execute(sitk_image, sitk_mask))
    except Exception as exc:
        return {"__error__": str(exc)}


def feature_class(feature):
    parts = feature.split("_")
    return parts[1] if len(parts) > 2 else ""


def symmetric_relative_error(manual, auto, eps=1e-8):
    return 2.0 * np.abs(auto - manual) / (np.abs(auto) + np.abs(manual) + eps)


def bland_altman(manual, auto):
    manual = np.asarray(manual, dtype=float)
    auto = np.asarray(auto, dtype=float)
    mean_value = (manual + auto) / 2.0
    diff = auto - manual
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
    lower = bias - 1.96 * sd
    upper = bias + 1.96 * sd
    if len(diff) >= 3 and np.std(mean_value, ddof=1) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(mean_value, diff)
    else:
        slope, p_value = np.nan, np.nan
    return bias, lower, upper, float(slope), float(p_value)


def corr_or_nan(x, y, method):
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    if method == "spearman":
        r, p = stats.spearmanr(x, y, nan_policy="omit")
    else:
        r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def icc2_for_feature(feature_df):
    long_rows = []
    for _, row in feature_df.iterrows():
        long_rows.append({"sample_id": row["sample_id"], "method": "manual", "value": float(row["manual"])})
        long_rows.append({"sample_id": row["sample_id"], "method": "auto", "value": float(row["auto"])})
    long_df = pd.DataFrame(long_rows)
    result = pg.intraclass_corr(
        data=long_df,
        targets="sample_id",
        raters="method",
        ratings="value",
        nan_policy="omit",
    )
    icc2_rows = result[result["Type"].isin(["ICC2", "ICC(A,1)"])]
    row = icc2_rows.iloc[0]
    ci_column = "CI95%" if "CI95%" in result.columns else "CI95"
    ci = row[ci_column]
    return float(row["ICC"]), float(ci[0]), float(ci[1]), float(row["F"]), float(row["pval"])


def summarize_feature_group(df, group, group_value):
    rows = []
    for feature, feature_df in df.groupby("feature"):
        feature_df = feature_df.dropna(subset=["manual", "auto"]).copy()
        if len(feature_df) < 3:
            continue
        manual = feature_df["manual"].to_numpy(dtype=float)
        auto = feature_df["auto"].to_numpy(dtype=float)
        try:
            icc, ci_low, ci_high, icc_f, icc_p = icc2_for_feature(feature_df)
        except Exception:
            icc, ci_low, ci_high, icc_f, icc_p = np.nan, np.nan, np.nan, np.nan, np.nan
        spearman_r, spearman_p = corr_or_nan(manual, auto, "spearman")
        pearson_r, pearson_p = corr_or_nan(manual, auto, "pearson")
        sre = symmetric_relative_error(manual, auto)
        bias, lower_loa, upper_loa, prop_slope, prop_p = bland_altman(manual, auto)
        q25, q75 = np.nanpercentile(sre, [25, 75])
        rows.append(
            {
                "group": group,
                "group_value": group_value,
                "feature": feature,
                "feature_class": feature_class(feature),
                "n_pairs": len(feature_df),
                "manual_mean": float(np.nanmean(manual)),
                "manual_std": float(np.nanstd(manual, ddof=1)),
                "auto_mean": float(np.nanmean(auto)),
                "auto_std": float(np.nanstd(auto, ddof=1)),
                "ICC2": icc,
                "ICC2_CI95_low": ci_low,
                "ICC2_CI95_high": ci_high,
                "ICC_F": icc_f,
                "ICC_p": icc_p,
                "spearman_rho": spearman_r,
                "spearman_p": spearman_p,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "median_symmetric_relative_error": float(np.nanmedian(sre)),
                "IQR_symmetric_relative_error": float(q75 - q25),
                "BA_bias": bias,
                "BA_lower_LoA": lower_loa,
                "BA_upper_LoA": upper_loa,
                "proportional_bias_slope": prop_slope,
                "proportional_bias_p": prop_p,
            }
        )
    if rows:
        p_values = [row["spearman_p"] if not np.isnan(row["spearman_p"]) else 1.0 for row in rows]
        _, fdr, _, _ = multipletests(p_values, method="fdr_bh")
        for row, p_adj in zip(rows, fdr):
            row["spearman_fdr_p"] = float(p_adj)
    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.ultralytics_root.is_dir():
        sys.path.insert(0, str(args.ultralytics_root))

    from ultralytics import YOLO

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = build_extractor(args.output_dir)

    image_paths = sorted((args.dataset_root / "images" / "val").glob("*"))
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
        image = read_image(image_path)
        h, w = image.shape
        patient_id, sequence, plane, image_id = parse_stem(image_path.stem)
        manual_mask = label_mask(label_dir / f"{image_path.stem}.txt", h, w)
        auto_mask = prediction_mask(result, h, w)
        manual_features = extract_features(extractor, image, manual_mask)
        auto_features = extract_features(extractor, image, auto_mask)

        sample_rows.append(
            {
                "sample_id": image_path.stem,
                "patient_id": patient_id,
                "sequence": sequence,
                "plane": plane,
                "image_id": image_id,
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
    for (group, group_value, cls), subset in pd.DataFrame(stat_rows).groupby(["group", "group_value", "feature_class"]):
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

    print(f"Saved {len(feature_rows)} feature rows")
    print(f"Saved {len(stat_rows)} agreement rows")
    print(f"Output {args.output_dir}")
    for row in summary_rows:
        if row["group"] == "overall":
            print(
                f"{row['feature_class']}: features={row['features']}, "
                f"median_ICC={row['median_ICC']:.3f}, stable={row['stable_features']}"
            )


if __name__ == "__main__":
    main()
