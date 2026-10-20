# PolyU Campus Image Classification (COMP4423 Assignment 2)

Classifies a scene photograph as taken **inside** or **outside** the PolyU main campus.
The classifier is a traditional machine-learning pipeline (hand-crafted features + SVM / Random Forest /
Logistic Regression, no deep learning) and only ever sees image pixels: never GPS, EXIF, file names or folders.

## Contents

1. [Quick start](#quick-start)
2. [Using the local web application](#using-the-local-web-application)
3. [Project layout](#project-layout)
4. [Dataset](#dataset)
5. [Training and evaluation](#training-and-evaluation)
6. [Experiments and diagnostic tools](#experiments-and-diagnostic-tools)
7. [Tests](#tests)
8. [Known limitations](#known-limitations)

## Quick start

Requires Python 3.10 or newer (developed with Python 3.13). Run every command from the project root.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # exact versions used: requirements-lock.txt
```

Start the web application with the final model:

```bash
python -m src.app --outputs-dir models
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). The final model is `models/final/pipeline.joblib`
(the app searches the folder given by `--outputs-dir` for every `pipeline.joblib`).

## Using the local web application

`python -m src.app` starts a server that listens on `127.0.0.1` only (nothing leaves your computer).
Options: `--port 8080`, `--no-browser`, `--outputs-dir <folder with pipeline.joblib files>`, `--dataset-dir dataset_dev`.

**Predict page**

- *Upload images*: choose or drag in one or many photos. Each photo is shown as a thumbnail with its own result,
  **"Inside PolyU main campus"** or **"Outside PolyU main campus"**, plus the decision score. A broken file is
  reported without stopping the others. *Export CSV* saves all results; *Clear all* resets the list.
- *Live camera*: press *Start camera* and allow camera access. One frame every 0.5, 1 or 2 seconds is classified;
  the page shows the latest result and a majority vote over the last 5 frames. The camera only works on
  `http://127.0.0.1:8000` or `http://localhost:8000`.

**Config page**

- Choose which saved pipeline to use (all `pipeline.joblib` files under `--outputs-dir`; the choice is remembered).
- See the pipeline steps with their parameters and the saved training metadata.
- See the dataset: images and location groups per class and split, every location group, and a warning if
  `images.csv` changed after the active model was trained.

The server passes only the raw image bytes to the saved scikit-learn pipeline, so the **same preprocessing and
feature extraction as in training and evaluation** is used. File names are never sent or used.

A terminal alternative for live prediction: `python main.py camera --model models/final/pipeline.joblib`.

## Project layout

```
main.py                 unified entry point: test | train | all | camera
src/
  app.py                local web application (python -m src.app)
  application/          command-line orchestration used by main.py
  camera/               live camera command-line tool
  data/                 dataset catalogue, validation, constants
  preprocessing/        decode, EXIF orientation, resize (squash / center_crop / pad)
  features/             colour histogram, HOG, optional LBP, scikit-learn feature transformer
  training/             classifier factory, grid search, metrics, model trainer
  storage/ utils/       atomic file writing, hashing, serialisation helpers
  experiments/          Logistic-Regression hold-out and feature-ablation experiments
  train.py              training command line (python -m src.train)
webui/                  HTML/JS/CSS of the web application
scripts/                dataset validation and splitting, experiment runners
tools/                  diagnostic helpers (variants, nearest neighbours, source audit, ...)
configs/                frozen experiment configurations
tests/                  all unit tests (python main.py test)
dataset_dev/            images.csv, locations.csv, split_report.json (photos are not kept in git)
models/                 trained pipelines (final model: models/final/)
```

## Quick Look

```bash
# 1. 掃描資料夾，更新 images.csv（新照片的 split 會是空的）
python scripts/update_images_csv.py --dataset-dir dataset_dev

# 2. 檢查一致性
python scripts/validate_dataset.py --dataset-dir dataset_dev

# 3. 預覽切分，再正式切分（已有切分要加 --force）
python scripts/split_dataset.py --dataset-dir dataset_dev --force --dry-run
python scripts/split_dataset.py --dataset-dir dataset_dev --force

# 4. 最後檢查，然後訓練
python scripts/validate_dataset.py --dataset-dir dataset_dev --require-split --require-verified
python main.py train --dataset-dir dataset_dev ...
```

## Dataset

The photographs are our own. Each photo belongs to a **location group** (nearby views of the same place,
including repeat visits). Folder layout: `images/<inside|outside>/<group_id>/<file>`; group ids are `in_001...`
and `out_001...`.

- `images.csv`: `image_id, relative_path, label (1 inside, 0 outside), group_id, session_id, scene_type, lighting, split`
- `locations.csv`: `group_id, class_name, location_description, boundary_verified`
- `split_report.json`: how the split was produced (seed, group counts, warnings)

Splits are made **by whole location group, separately inside each class**, aiming for 60% train / 20% validation /
20% test of the groups, so no scene appears in two splits. The exact image and group counts per class and split are on
the Config page of the web application and in `split_report.json`.

```bash
python scripts/validate_dataset.py --dataset-dir dataset_dev --require-split   # read-only consistency check
python scripts/split_dataset.py --dataset-dir dataset_dev --dry-run             # preview a split
python scripts/split_dataset.py --dataset-dir dataset_dev                       # write the split (refuses if one exists)
```

An existing split is frozen: `split_dataset.py` refuses to overwrite it unless `--force` is given. Never redraw it
after looking at test results. Before the final run every group must be checked against the official PolyU campus map and
marked `boundary_verified = yes` in `locations.csv`; unverified groups are rejected unless `--allow-unverified` is
used, and such runs are labelled as development runs.

## Training and evaluation

```bash
python main.py train --dataset-dir dataset_dev --models-dir models/final --results-dir results/final --resize-mode center_crop
python main.py train --help        # all options: feature sets, classifiers, image size, HOG cell, seed, ...
```

- Preprocessing: decode, apply EXIF orientation, convert to RGB, resize to a square (`--resize-mode squash | center_crop | pad`).
- Features: colour histogram, HOG, optionally LBP (`--feature-sets`).
- Classifiers: chosen by `--classifiers`; class weights are balanced unless `--no-class-weight`.
- Model selection uses **validation macro-F1** only; ties are broken by balanced accuracy, then accuracy.
- The **test split is never opened by training**. Evaluate it once, after the configuration is frozen.
- Outputs: `pipeline.joblib`, `pipeline_metadata.json` (configuration, counts, seed, hash of `images.csv`), and in the
  results folder `validation_experiments.csv` and `training_report.json`.
- Reproducibility: fixed seed (`--seed`, default 42), pinned packages in `requirements-lock.txt`.

`python main.py all ...` runs the tests first and trains only if they pass.

## Experiments and diagnostic tools

```bash
python scripts/run_polyu_lr.py --config configs/polyu_lr_01.json --dry-run          # Logistic Regression hold-out experiment
python scripts/prepare_polyu_feature_experiment.py --freeze                         # freeze the feature-ablation plan
python scripts/run_feature_experiment.py --dry-run                                  # verify the frozen plan, then run without --dry-run
```

Helpers in `tools/` (diagnostics only; their outputs are never added to the dataset):


| Tool                                        | Purpose                                                                                       |
| ------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `make_variants.py`, `summarise_variants.py` | single-factor test variants (grayscale, colour swap, blur, crop, ...) and a flip-rate summary |
| `nearest_train.py`                          | training images that look most like a given photo in feature space                            |
| `audit_sources.py`                          | compare resolution, compression and sharpness between the two classes                         |
| `find_duplicates.py`                        | near-duplicate photos inside and between location groups                                      |
| `preview_modes.py`                          | what the model receives under each resize mode                                                |
| `organise_photos.py`                        | add newly taken photos to the dataset layout from an editable manifest                        |
| `consolidate_tests.py`                      | one-off helper that moved all test packages under`tests/`                                     |

## Tests

```bash
python main.py test                              # all unit tests
python -m unittest discover -s tests -t . -v     # same, directly
```

## Known limitations

- The dataset is small (a few dozen location groups), so validation and test results depend on a handful of scenes.
  Report results per location group as well as overall.
- Hand-crafted colour and HOG features can latch onto scene-type shortcuts, for example PolyU's red-brick facades or
  regular window grids, so building facades outside the campus can be classified as inside.
- Photos from different cameras or exports differ in resolution, compression and aspect ratio; the resize mode
  matters (see `tools/preview_modes.py`).
- Classification uses only the picture. Do not treat a prediction as proof of a real location.
