from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ultralytics-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--include-zero-dice", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--one-by-one", action="store_true")
    return parser.parse_args()


def collect_images(split_paths):
    from ultralytics.data.utils import IMG_FORMATS

    paths = split_paths if isinstance(split_paths, list) else [split_paths]
    images = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            images.extend(p for p in path.rglob("*") if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS)
        elif path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                item = line.strip()
                if not item:
                    continue
                image = Path(item)
                if item.startswith("./") or item.startswith(".\\"):
                    image = path.parent / item[2:]
                images.append(image)
        else:
            raise FileNotFoundError(f"Validation path does not exist: {path}")

    images = sorted({str(p.resolve()) for p in images if p.suffix[1:].lower() in IMG_FORMATS})
    if not images:
        raise FileNotFoundError(f"No validation images found in: {split_paths}")
    return images


def ground_truth_mask(label_path, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file does not exist: {label_path}")

    count = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"Invalid segmentation polygon at {label_path}:{line_number}")
        coordinates = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
        coordinates[:, 0] = np.clip(coordinates[:, 0] * width, 0, width - 1)
        coordinates[:, 1] = np.clip(coordinates[:, 1] * height, 0, height - 1)
        cv2.fillPoly(mask, [coordinates.astype(np.int32)], color=1)
        count += 1
    return mask, count


def predicted_union(result, height, width):
    if result.masks is None or len(result.masks.data) == 0:
        return np.zeros((height, width), dtype=np.uint8), 0
    masks = result.masks.data.detach().cpu().numpy()
    union = np.any(masks > 0.5, axis=0).astype(np.uint8)
    if union.shape != (height, width):
        union = cv2.resize(union, (width, height), interpolation=cv2.INTER_NEAREST)
    return union, len(masks)


def score_masks(prediction, target):
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    pred_pixels = int(pred.sum())
    gt_pixels = int(gt.sum())
    union = pred_pixels + gt_pixels - intersection
    dice_denominator = pred_pixels + gt_pixels
    dice = 2.0 * intersection / dice_denominator if dice_denominator else 1.0
    iou = intersection / union if union else 1.0
    return intersection, pred_pixels, gt_pixels, union, dice, iou


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def main():
    args = parse_args()
    if args.ultralytics_root.is_dir():
        sys.path.insert(0, str(args.ultralytics_root))

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset, img2label_paths

    dataset = check_det_dataset(str(args.data), autodownload=False)
    image_paths = collect_images(dataset["val"])
    label_paths = img2label_paths(image_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ultralytics: {YOLO.__module__}")
    print(f"Weights: {args.weights}")
    print(f"Dataset: {args.data}")
    print(f"Validation images: {len(image_paths)}")

    model = YOLO(str(args.weights))
    rows = []
    totals = {"intersection": 0, "pred_pixels": 0, "gt_pixels": 0, "union": 0}
    if args.one_by_one:
        result_iter = (
            model.predict(
                source=image_path,
                imgsz=args.imgsz,
                batch=1,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                retina_masks=True,
                stream=False,
                verbose=False,
            )[0]
            for image_path in image_paths
        )
    else:
        result_iter = model.predict(
            source=image_paths,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            retina_masks=True,
            stream=True,
            verbose=False,
        )

    for index, result in enumerate(result_iter):
        image_path = Path(image_paths[index])
        label_path = Path(label_paths[index])
        height, width = result.orig_shape
        gt_mask, gt_instances = ground_truth_mask(label_path, height, width)
        pred_mask, pred_instances = predicted_union(result, height, width)
        intersection, pred_pixels, gt_pixels, union, dice, iou = score_masks(pred_mask, gt_mask)
        included = args.include_zero_dice or dice != 0.0
        if included:
            totals["intersection"] += intersection
            totals["pred_pixels"] += pred_pixels
            totals["gt_pixels"] += gt_pixels
            totals["union"] += union
        rows.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "gt_instances": gt_instances,
                "pred_instances": pred_instances,
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "intersection": intersection,
                "union": union,
                "dice": dice,
                "iou": iou,
                "included_in_summary": included,
            }
        )
        if not args.quiet:
            print(f"[{index + 1}/{len(image_paths)}] {image_path.name}: Dice={dice:.6f} IoU={iou:.6f}")
        if args.one_by_one and (index + 1) % 100 == 0:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    included_rows = [row for row in rows if row["included_in_summary"]]
    excluded_rows = [row for row in rows if not row["included_in_summary"]]
    micro_dice_denominator = totals["pred_pixels"] + totals["gt_pixels"]
    summary = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "split": "val",
        "images": len(rows),
        "evaluated_images": len(included_rows),
        "excluded_images": [Path(row["image"]).name for row in excluded_rows],
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "conf": args.conf,
            "nms_iou": args.iou,
            "max_det": args.max_det,
            "include_zero_dice": args.include_zero_dice,
        },
        "macro_dice": mean_or_none([row["dice"] for row in included_rows]),
        "macro_iou": mean_or_none([row["iou"] for row in included_rows]),
        "micro_dice": 2.0 * totals["intersection"] / micro_dice_denominator if micro_dice_denominator else 1.0,
        "micro_iou": totals["intersection"] / totals["union"] if totals["union"] else 1.0,
        "totals": totals,
    }

    csv_path = args.output_dir / "per_image_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nValidation mask metrics")
    print(f"Included images: {len(included_rows)}/{len(rows)}")
    print(f"Excluded images: {', '.join(summary['excluded_images']) if excluded_rows else 'None'}")
    print(f"Macro Dice: {summary['macro_dice']:.6f}")
    print(f"Macro IoU:  {summary['macro_iou']:.6f}")
    print(f"Micro Dice: {summary['micro_dice']:.6f}")
    print(f"Micro IoU:  {summary['micro_iou']:.6f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
