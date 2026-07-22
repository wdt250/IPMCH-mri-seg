from __future__ import annotations

import argparse

from ultralytics.models.yolo.segment.train_proto import DeBiFormerProtoTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeBiFormer DSAM segmentation with cross-plane prototype loss.")
    parser.add_argument("--data", required=True, help="Path to the Ultralytics dataset YAML.")
    parser.add_argument("--model", default="ultralytics/cfg/models/11/yolo11-seg-dsam.yaml", help="Model YAML or weights.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=0)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    if args.batch % 3 != 0:
        raise ValueError("For triplet training, --batch should be divisible by 3.")

    return args


def main() -> None:
    args = parse_args()
    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "overlap_mask": True,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "multi_scale": False,
    }

    if args.project is not None:
        overrides["project"] = args.project
    if args.name is not None:
        overrides["name"] = args.name

    trainer = DeBiFormerProtoTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
