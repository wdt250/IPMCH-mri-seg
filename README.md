# IPMCH-mri-seg

Research code for efficient multiplanar placental MRI segmentation and manual-versus-automatic ROI radiomics agreement analysis.

This repository extends Ultralytics YOLO11 segmentation with DeBiFormerPlus attention and a cross-plane foreground-prototype consistency objective. A triplet sampler keeps sagittal, coronal and transverse images from the same participant and MRI sequence in one single-GPU batch.

## Scope

- DeBiFormerPlus-enhanced YOLO11 segmentation.
- Cross-plane prototype consistency loss.
- Participant- and sequence-aware sagittal/coronal/transverse sampling.
- Dice and IoU evaluation for YOLO polygon masks.
- DICOM-space PyRadiomics agreement analysis.
- ICC confidence intervals, correlation, FDR correction, symmetric relative error and Bland-Altman summaries.

The clinical images, DICOM files, annotations and trained weights are not included. They may contain protected information or are subject to institutional data-use restrictions.

## Repository layout

```text
IPMCH-mri-seg/
|-- train_debiformer_proto.py
|-- configs/
|   |-- placenta_dataset.example.yaml
|   `-- dicom_match.example.csv
|-- research/
|   |-- build_triplet_dataset.py
|   |-- evaluate_segmentation.py
|   |-- compute_radiomics_agreement.py
|   |-- compute_radiomics_agreement_jpg.py
|   |-- analyze_icc_sensitivity.py
|   |-- plot_radiomics_agreement.py
|   `-- summarize_model_radiomics.py
`-- ultralytics/
    |-- data/triplet_sampler.py
    |-- models/yolo/segment/train_proto.py
    |-- nn/attention/DSAM.py
    |-- utils/plane_consistency.py
    `-- cfg/models/11/yolo11-seg-dsam.yaml
```

## Installation

Python 3.10 was used for the reported experiments.

```bash
git clone https://github.com/wdt250/IPMCH-mri-seg.git
cd IPMCH-mri-seg

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install a CUDA-enabled PyTorch build suitable for the local driver first.
pip install -e .
pip install -r requirements-research.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Dataset format

Images and YOLO polygon labels must use the same stem. The cross-plane sampler expects this de-identified filename convention:

```text
<case>_<sequence>-<plane>_<image_id>.jpg
```

Example:

```text
case001_haste-sag_0001.jpg
case001_haste-cor_0002.jpg
case001_haste-tra_0003.jpg
```

The supported plane codes are `sag`, `cor` and `tra`. The triplet key is `<case>_<sequence>`. Participant-level train/validation separation can be generated with:

```bash
python research/build_triplet_dataset.py \
  --images /path/to/source/images \
  --labels /path/to/source/labels \
  --output /path/to/placenta_triplets \
  --train-ratio 0.9 \
  --class-name placenta
```

Use `--overwrite` only when the existing output directory can be replaced.

## Training

Edit `configs/placenta_dataset.example.yaml` so that `path` points to the prepared dataset. The primary experiment used 150 epochs, 640-pixel input and a batch size of 12:

```bash
python train_debiformer_proto.py \
  --data configs/placenta_dataset.example.yaml \
  --model ultralytics/cfg/models/11/yolo11-seg-dsam.yaml \
  --epochs 150 \
  --imgsz 640 \
  --batch 12 \
  --workers 4 \
  --device 0 \
  --project runs \
  --name debiformerplus_consistency_150ep
```

The batch size must be divisible by three. The current triplet loader supports single-GPU training. Mosaic, MixUp, copy-paste, horizontal/vertical flipping and multiscale training are disabled by the entry script to preserve MRI orientation and complete triplets.

## Segmentation evaluation

```bash
python research/evaluate_segmentation.py \
  --weights runs/debiformerplus_consistency_150ep/weights/best.pt \
  --data configs/placenta_dataset.example.yaml \
  --output-dir outputs/segmentation_metrics \
  --device 0
```

The script writes per-image and aggregate Dice/IoU results. No weights are downloaded automatically.

## DICOM-space radiomics agreement

The matching CSV must contain at least `image`, `split`, `metadata_match` and `best_dicom_path`. See `configs/dicom_match.example.csv`.

```bash
python research/compute_radiomics_agreement.py \
  --dataset-root /path/to/placenta_triplets \
  --match-csv /path/to/deidentified_dicom_match.csv \
  --weights runs/debiformerplus_consistency_150ep/weights/best.pt \
  --output-dir outputs/radiomics_agreement \
  --device 0
```

The analysis extracts original-image shape2D, first-order, GLCM, GLRLM, GLSZM, GLDM and NGTDM features with identical settings for manual and automatic ROIs. The exact extraction settings are provided as a standalone PyRadiomics parameter file at `configs/pyradiomics_params.yaml` (the same structure is also written to `pyradiomics_params.json` in each output directory). Stable features are selected with all of the following criteria:

```text
ICC(A,1) >= 0.90
Spearman rho >= 0.80
FDR-adjusted p < 0.05
median symmetric relative error <= 0.20
```

The stricter output additionally requires the lower bound of the ICC 95% confidence interval to be at least 0.85.

Participant-level sensitivity analysis and optional QC visualization can then be run with:

```bash
python research/analyze_icc_sensitivity.py \
  --dataset-root /path/to/placenta_triplets \
  --result-dir outputs/radiomics_agreement \
  --weights runs/debiformerplus_consistency_150ep/weights/best.pt \
  --device 0
```

Add `--qc-sample <deidentified_image_stem>` to generate an overlay for one selected sample.

## Reproducibility notes

- Ultralytics base version: 8.3.158.
- Primary training duration: 150 epochs.
- Default model scale: `n` with 3,130,301 parameters in the current configuration.
- Input size: 640 x 640.
- Training batch size: 12, corresponding to four complete plane triplets.
- Prototype consistency weight: 0.02.
- Segmentation confidence threshold: 0.25.
- Non-maximum suppression IoU threshold: 0.70.

Exact results also depend on the GPU, CUDA/PyTorch build, data split and image preprocessing. The original clinical dataset is not publicly redistributed.

## License and attribution

This repository is derived from Ultralytics and remains under the GNU Affero General Public License v3.0. See `LICENSE` and `README.upstream.md`. Ultralytics trademarks and upstream copyrights remain with their respective owners.

## Citation

The associated manuscript citation will be added after publication. Until then, cite this repository and the upstream methods used in your work.
