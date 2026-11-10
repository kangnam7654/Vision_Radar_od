# Radar Object Detection

[한국어](./README_ko.md)

> **Portfolio Project** — A computer vision project for detecting objects on marine radar screens.

## Overview

This project detects objects such as vessels in real-time from radar imagery.
While radar UIs vary across devices, the project handles multiple radar types by converting raw LabelMe annotations to a unified YOLO format with stratified sampling, then performs detection with a YOLO model.

### Data

Two types of radar data are used. Although the UIs differ, the circular scan area shares the same structure.

| Radar Type 1 | Radar Type 2 |
|:---:|:---:|
| ![radar1](./sample_images/radar1.png) | ![radar2](./sample_images/radar2.png) |

### Detection Classes

| Class | Description |
|-------|-------------|
| `boat_rectangle` | Rectangular vessel |
| `boat_triangle` | Triangular vessel |
| `start` | Start marker |
| `end` | End marker |

## Method

The preprocessing pipeline converts raw annotated data into a YOLO-compatible dataset:

### 1. Annotation Conversion

LabelMe JSON annotations are converted to YOLO format. Class labels are standardized (typo correction, name unification) and bounding box coordinates are normalized to `(class_id, x_center, y_center, width, height)`.

### 2. Stratified Split

Data is split into train/valid/test sets (default 80/10/10) with stratified sampling by radar type, ensuring each split has balanced representation from all radar sources.

## Result

![result](./sample_images/result.png)

## Project Structure

```
.
├── src/
│   ├── train_model.py                          # Training pipeline
│   ├── inference.py                            # Inference pipeline
│   ├── yolo26n.pt                              # Pre-trained YOLO26 Nano weights
│   ├── yolo26m.pt                              # Pre-trained YOLO26 Medium weights
│   ├── yaml/
│   │   ├── radar_config.yaml                   # Training/inference/preprocessing config
│   │   └── radar_dataset.yaml                  # Dataset definition (auto-updated)
│   ├── utils/
│   │   ├── train/
│   │   │   └── pre_process_modules.py          # Preprocessing (LabelMe→YOLO conversion, stratified split)
│   │   ├── inference/
│   │   │   └── inference_modules.py            # Inference modules (model loading, coordinate inverse-transform)
│   │   └── common/
│   │       ├── project_paths.py                # Path utilities
│   │       ├── video_frame_extractor.py        # Video frame extraction for data collection
│   │       ├── random_rotate.py                # Random rotation augmentation
│   │       └── save_json.py                    # JSON saving
│   └── test_inference/
│       ├── test_inference.py                   # Batch inference script over test_data (not a unit test)
│       ├── test_circle_coordinates_old.json    # Scan-circle pixel coordinates used by the sample demo
│       ├── test_data/                          # Test images
│       └── test_results/                       # Historical outputs of an earlier model (unchanged)
├── tests/
│   └── test_inference_logic.py                 # Deterministic unittest suite (synthetic images, fake model)
├── outputs/                                    # Inference outputs (git-ignored)
├── data/                                       # Raw & processed data (git-ignored)
│   ├── radar1/                                 # Raw radar type 1 (PNG + LabelMe JSON)
│   ├── radar2/                                 # Raw radar type 2
│   └── train_data/raw_stratified/              # Processed YOLO dataset (train/valid/test)
├── runs/detect/                                # Training run outputs (weights, metrics, plots)
├── sample_images/                              # README images
├── pyproject.toml
└── README.md
```

## Setup

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Requirements

- Python >= 3.11
- CUDA GPU (recommended for training)

## Usage

### Preprocessing

Converts raw LabelMe annotations into a YOLO-format dataset with stratified train/valid/test split.

```bash
# Run preprocessing standalone
uv run python src/utils/train/pre_process_modules.py
```

This reads from `data/radar1/` and `data/radar2/`, and outputs to `data/train_data/raw_stratified/`.

### Training

```bash
uv run python src/train_model.py
```

By default, preprocessing is skipped (assumes data is already prepared). To run preprocessing before training, edit `src/train_model.py` and set `main(run_pre_process=True)`.

Training configuration is in `src/yaml/radar_config.yaml`:

```yaml
train_conf:
  model: yolo26n.pt       # YOLO model (yolo26n.pt or yolo26m.pt)
  epochs: 1000
  batch_size: 512
  image_size: 320
  device: 0               # GPU ID or cpu
  patience: 100           # Early stopping patience
```

### Inference (public sample demo)

Run from the repository root:

```bash
uv run python src/inference.py
```

With the default config this processes the single public sample `sample_images/radar2.png` using only files already committed to this repository:

| Input | Path |
|-------|------|
| Image | `sample_images/radar2.png` |
| Checkpoint | `runs/detect/train4/weights/best.pt` (Ultralytics YOLO26n fine-tune, run `train4`, see `runs/detect/train4/args.yaml`) |
| Scan circle | `src/test_inference/test_circle_coordinates_old.json` (`x_centre=438, y_centre=348, radius=339`, pixels) |

Outputs are written to the git-ignored `outputs/inference/` folder:

- `radar2.png.annotated.png` — the original-resolution image with boxes drawn on a copy
- `radar2.png.detections.json` — sidecar JSON (always valid, `"detections": []` when nothing is found)

Pipeline: crop the circle's bounding square (must lie fully inside the image, otherwise an error is raised), zero the pixels outside the scan circle, run the model on the crop, then map boxes back to the original image. In the JSON, `xyxy_px` are **absolute pixel** coordinates `(x1, y1, x2, y2)` in the original image (crop offset added, clipped to the image bounds) and `crop_xyxy_px` are pixel coordinates inside the crop; normalized 0–1 ratios are not stored. Each detection keeps `class_id`, `class_name` and `confidence`.

Configuration lives in `src/yaml/radar_config.yaml` (relative paths are resolved from the repository root; absolute paths are accepted):

```yaml
inference:
  source: sample_images/radar2.png          # image file or folder
  weight: runs/detect/train4/weights/best.pt
  circle_json: src/test_inference/test_circle_coordinates_old.json  # flat, or nested under "old_radar"
  save: true
  save_dir: outputs/inference
  show: false                               # opt-in preview window
  conf: 0.25
  grayscale: false                          # true = legacy gray->BGR model input
```

CLI overrides: `--source`, `--weights`, `--circle-json`, `--output-dir`, `--conf`, `--config`, `--show`, `--no-save`.

Headless behaviour: no window is opened by default, so it works over SSH/CI. `--show` is opt-in and requires a GUI-capable OpenCV build plus a display; saving does not depend on it. Missing files, an unreadable/empty image, an empty input folder, or an out-of-frame circle produce an error message and exit code 2. A missing checkpoint is **never** downloaded or retrained.

Limitations:

- This is a demonstration on one public sample image only. The circle JSON is specific to that radar layout/resolution; other images need their own circle coordinates.
- The committed checkpoint is a YOLO26 model. It is not the earlier YOLOv5 model described in older portfolio material and is not claimed to reproduce that project's results; no accuracy metrics are claimed here.
- `src/test_inference/test_results/` holds historical outputs from an earlier model with different classes and is kept unchanged for reference. `uv run python src/test_inference/test_inference.py` re-runs the current pipeline on `test_data/` into `outputs/test_inference/` instead.

### Tests

```bash
uv run python -m unittest discover -s tests -v
```

These are deterministic unit tests for path resolution, circle validation/cropping, pixel coordinate mapping/clipping and saving. They use temporary synthetic images and a fake model output — they do **not** run the real model and say nothing about detection quality.

## Tech Stack

- **Object Detection**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- **Image Processing**: OpenCV
- **Labeling**: [LabelMe](https://github.com/wkentaro/labelme)
- **Package Management**: [uv](https://github.com/astral-sh/uv)
