import argparse
import os
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGES_DIR: Path
LABELS_DIR: Path
OUTPUT_DIR: Path
TRAIN_RATIO: float
SEED: int
COPY_MODE: str
CLASS_NAMES: list[str]

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PLANE_KEYS = ("sag", "cor", "tra")


@dataclass
class Candidate:
    patient_id: str
    sequence_name: str
    plane: str
    image_id: str
    image_name: str
    label_name: str
    seg_area: float


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def reset_output_dir(path: Path):
    target = path.resolve()
    if target == Path(target.anchor):
        raise ValueError(f"Refuse to reset a filesystem root: {target}")
    if path.exists():
        shutil.rmtree(path)


def transfer_file(src: Path, dst: Path, mode="copy"):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported COPY_MODE: {mode}")


def image_to_label_path(image_name: str):
    return LABELS_DIR / f"{Path(image_name).stem}.txt"


def parse_image_stem(stem: str):
    # Expected: case_sequence-plane_imageid, e.g. case001_haste-sag_0001
    try:
        left, image_id = stem.rsplit("_", 1)
        patient_id, sequence_plane = left.split("_", 1)
        sequence_name, plane = sequence_plane.rsplit("-", 1)
    except ValueError:
        return None

    plane = plane.lower()
    if plane not in PLANE_KEYS:
        return None

    return patient_id, sequence_name, plane, image_id


def polygon_area(points):
    if len(points) < 3:
        return 0.0

    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def label_seg_area(label_path: Path):
    total_area = 0.0

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            coords = parts[1:]
            coord_count = len(coords)
            if coord_count % 2 != 0:
                coord_count -= 1

            try:
                point_count = coord_count // 2
                if point_count < 3:
                    continue

                first_x = float(coords[0])
                first_y = float(coords[1])
                prev_x = first_x
                prev_y = first_y
                area = 0.0

                for idx in range(2, coord_count, 2):
                    x = float(coords[idx])
                    y = float(coords[idx + 1])
                    area += prev_x * y - x * prev_y
                    prev_x = x
                    prev_y = y

                area += prev_x * first_y - first_x * prev_y
            except ValueError:
                continue

            total_area += abs(area) / 2.0

    return total_area


def build_image_index():
    image_index = {}
    for image_path in IMAGES_DIR.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTS:
            image_index[image_path.stem] = image_path.name
    return image_index


def collect_max_area_triplets():
    image_index = build_image_index()
    best_by_key = {}
    errors = []

    for label_path in LABELS_DIR.glob("*.txt"):
        parsed = parse_image_stem(label_path.stem)
        if parsed is None:
            errors.append(["parse_label_name_failed", label_path.name])
            continue

        patient_id, sequence_name, plane, image_id = parsed
        image_name = image_index.get(label_path.stem)
        if image_name is None:
            errors.append(["missing_image_for_label", label_path.name])
            continue

        seg_area = label_seg_area(label_path)
        candidate = Candidate(
            patient_id=patient_id,
            sequence_name=sequence_name,
            plane=plane,
            image_id=image_id,
            image_name=image_name,
            label_name=label_path.name,
            seg_area=seg_area,
        )

        key = (patient_id, sequence_name, plane)
        old = best_by_key.get(key)
        if old is None or candidate.seg_area > old.seg_area:
            best_by_key[key] = candidate

    groups = []
    patient_sequence_keys = sorted({(k[0], k[1]) for k in best_by_key})
    for patient_id, sequence_name in patient_sequence_keys:
        plane_best = {
            plane: best_by_key.get((patient_id, sequence_name, plane))
            for plane in PLANE_KEYS
        }
        missing_planes = [plane for plane, item in plane_best.items() if item is None]
        if missing_planes:
            errors.append([
                "incomplete_planes",
                patient_id,
                sequence_name,
                "|".join(missing_planes),
            ])
            continue

        groups.append({
            "patient_id": patient_id,
            "sequence_name": sequence_name,
            "sag_file": plane_best["sag"].image_name,
            "cor_file": plane_best["cor"].image_name,
            "tra_file": plane_best["tra"].image_name,
            "sag_image_id": plane_best["sag"].image_id,
            "cor_image_id": plane_best["cor"].image_id,
            "tra_image_id": plane_best["tra"].image_id,
            "sag_seg_area": f"{plane_best['sag'].seg_area:.10f}",
            "cor_seg_area": f"{plane_best['cor'].seg_area:.10f}",
            "tra_seg_area": f"{plane_best['tra'].seg_area:.10f}",
        })

    return groups, errors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a participant-separated YOLO triplet dataset using the largest ROI per plane."
    )
    parser.add_argument("--images", type=Path, required=True, help="Directory containing source images.")
    parser.add_argument("--labels", type=Path, required=True, help="Directory containing YOLO polygon labels.")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--class-name", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def main():
    global IMAGES_DIR, LABELS_DIR, OUTPUT_DIR, TRAIN_RATIO, SEED, COPY_MODE, CLASS_NAMES

    args = parse_args()
    IMAGES_DIR = args.images.resolve()
    LABELS_DIR = args.labels.resolve()
    OUTPUT_DIR = args.output.resolve()
    TRAIN_RATIO = float(args.train_ratio)
    SEED = int(args.seed)
    COPY_MODE = args.copy_mode
    CLASS_NAMES = args.class_name or ["placenta"]

    if not 0.0 < TRAIN_RATIO < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        raise FileNotFoundError("Both --images and --labels must be existing directories.")
    for source_dir in (IMAGES_DIR, LABELS_DIR):
        if OUTPUT_DIR == source_dir or OUTPUT_DIR in source_dir.parents or source_dir in OUTPUT_DIR.parents:
            raise ValueError(f"Output directory must not overlap a source directory: {source_dir}")
    if OUTPUT_DIR.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {OUTPUT_DIR}. Pass --overwrite to replace it.")

    random.seed(SEED)
    reset_output_dir(OUTPUT_DIR)

    img_train_dir = OUTPUT_DIR / "images" / "train"
    img_val_dir = OUTPUT_DIR / "images" / "val"
    lab_train_dir = OUTPUT_DIR / "labels" / "train"
    lab_val_dir = OUTPUT_DIR / "labels" / "val"

    for d in [img_train_dir, img_val_dir, lab_train_dir, lab_val_dir]:
        safe_mkdir(d)

    groups, errors = collect_max_area_triplets()
    if not groups:
        raise ValueError("No complete sag/cor/tra triplets were found from labels.")

    # Split by patient_id so different sequences from the same patient
    # never appear in both train and val.
    patient_ids = sorted({group["patient_id"] for group in groups})
    random.shuffle(patient_ids)

    n_total = len(groups)
    n_train_patients = int(round(len(patient_ids) * TRAIN_RATIO))
    train_patient_ids = set(patient_ids[:n_train_patients])
    val_patient_ids = set(patient_ids[n_train_patients:])

    train_groups = [
        group for group in groups
        if group["patient_id"] in train_patient_ids
    ]
    val_groups = [
        group for group in groups
        if group["patient_id"] in val_patient_ids
    ]

    split_rows = []

    def process_group(group, split):
        if split == "train":
            img_out_dir = img_train_dir
            lab_out_dir = lab_train_dir
        else:
            img_out_dir = img_val_dir
            lab_out_dir = lab_val_dir

        for plane_key in ["sag_file", "cor_file", "tra_file"]:
            img_name = group[plane_key]
            img_src = IMAGES_DIR / img_name
            lab_src = image_to_label_path(img_name)

            if not img_src.exists():
                errors.append([split, "missing_image", str(img_src)])
                continue
            if not lab_src.exists():
                errors.append([split, "missing_label", str(lab_src)])
                continue

            transfer_file(img_src, img_out_dir / img_src.name, COPY_MODE)
            transfer_file(lab_src, lab_out_dir / lab_src.name, COPY_MODE)

        split_rows.append({
            "patient_id": group["patient_id"],
            "sequence_name": group["sequence_name"],
            "split": split,
            "sag_file": group["sag_file"],
            "cor_file": group["cor_file"],
            "tra_file": group["tra_file"],
            "sag_image_id": group["sag_image_id"],
            "cor_image_id": group["cor_image_id"],
            "tra_image_id": group["tra_image_id"],
            "sag_seg_area": group["sag_seg_area"],
            "cor_seg_area": group["cor_seg_area"],
            "tra_seg_area": group["tra_seg_area"],
        })

    for g in train_groups:
        process_group(g, "train")

    for g in val_groups:
        process_group(g, "val")

    split_csv = OUTPUT_DIR / "triplet_split.csv"
    split_fieldnames = [
        "patient_id", "sequence_name", "split",
        "sag_file", "cor_file", "tra_file",
        "sag_image_id", "cor_image_id", "tra_image_id",
        "sag_seg_area", "cor_seg_area", "tra_seg_area",
    ]
    with open(split_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=split_fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)

    error_csv = OUTPUT_DIR / "build_errors.csv"
    with open(error_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_type", "field_1", "field_2", "field_3"])
        writer.writerows(errors)

    yaml_path = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: .\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {name}\n")

    train_image_count = len(list(img_train_dir.glob("*")))
    val_image_count = len(list(img_val_dir.glob("*")))
    train_label_count = len(list(lab_train_dir.glob("*.txt")))
    val_label_count = len(list(lab_val_dir.glob("*.txt")))

    print("=" * 70)
    print("Build YOLO11 triplet dataset by max segmentation area")
    print("Triplet key: patient_id + sequence_name")
    print("Split key: patient_id")
    print(f"Total complete triplets: {n_total}")
    print(f"Total patients: {len(patient_ids)}")
    print(f"Train patients: {len(train_patient_ids)}")
    print(f"Val patients: {len(val_patient_ids)}")
    print(f"Train triplets: {len(train_groups)}")
    print(f"Val triplets: {len(val_groups)}")
    print("-" * 70)
    print(f"Train images: {train_image_count}")
    print(f"Train labels: {train_label_count}")
    print(f"Val images: {val_image_count}")
    print(f"Val labels: {val_label_count}")
    print("-" * 70)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"split csv: {split_csv}")
    print(f"errors csv: {error_csv}")
    print(f"dataset.yaml: {yaml_path}")
    print(f"Errors: {len(errors)}")
    if errors:
        print("First 10 errors:")
        for e in errors[:10]:
            print(e)
    print("=" * 70)


if __name__ == "__main__":
    main()
