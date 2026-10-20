# PolyU Main Campus Image Classification

COMP4423 Computer Vision — Assignment 2
2026/27 Semester 1

A local application that classifies scene photographs as taken inside or outside the PolyU main campus using hand-crafted visual features and a traditional machine-learning classifier. **No deep learning is used.**

## 1. Task and Class Definitions

The class labels are:


| Label | Class   | Definition                                                                                    |
| ----- | ------- | --------------------------------------------------------------------------------------------- |
| 1     | inside  | The photograph was taken within the PolyU main campus and mainly depicts a scene within it.   |
| 0     | outside | The photograph was taken outside the PolyU main campus and mainly depicts a scene outside it. |

Labels describe geographic location, not whether a scene is indoors or outdoors.

Ground truth is checked against the official PolyU Campus Map. Red bricks, university logos and other visual similarities do not determine the label. Uncertain boundary locations, gate thresholds, connecting bridges and views across the campus boundary must be excluded according to the assignment rules.

The classifier uses image pixels only. GPS, location notes, filenames and folder names are not input features. EXIF orientation is used only to decode the image in the correct orientation.

## 2. Quick Start

Run all commands from the project root containing `main.py`, `src/` and `webui/`.

### 2.1 Create the environment

The saved final-model metadata records Python 3.13.7. Using the recorded Python and package versions is recommended for loading and reproducing the serialized model.

```
python -m venv .venv
```

Activate the environment on macOS or Linux:

```
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```
.venv\Scripts\Activate.ps1
```

Install the recorded package versions:

```
python -m pip install -r requirements-lock.txt
```

Alternatively, install the unpinned requirements:

```
python -m pip install -r requirements.txt
```

The unpinned requirements may install different package versions. They are not the preferred option for reproducing the saved model.

Only load trusted `joblib` files because deserialization can execute code.

### 2.2 Launch the application

```
python -m src.app --outputs-dir models/final --dataset-dir dataset_dev
```

Open the following address in a browser:

```
http://127.0.0.1:8000
```

Stop the server with `Ctrl+C`.

Using `models/final` as the search directory avoids selecting an experimental model accidentally. Starting the application does not retrain the model or automatically evaluate the test set.

If port 8000 is unavailable:

```
python -m src.app --outputs-dir models/final --dataset-dir dataset_dev --port 8080 --no-browser
```

Then open:

```
http://127.0.0.1:8080
```

## 3. Application Usage

### 3.1 Predict page

Upload or select an image. The application displays it and returns one of:

* `Inside PolyU main campus`
* `Outside PolyU main campus`

The application passes image bytes through the saved preprocessing, feature extraction, scaling and classification pipeline. A filename or class label is not required as prediction input.

For the final binary classifier:

* A positive decision score supports inside.
* A negative decision score supports outside.
* A decision score is not a probability.

The server rejects request bodies larger than 25 MiB.

The interface also supports live-camera prediction. Camera access is optional and requires browser permission.

### 3.2 Config page

The Config page displays:

* The active model.
* Pipeline steps and parameters.
* Saved training metadata.
* Dataset image and location-group counts.
* Location descriptions and boundary-verification records.
* A warning when the current `images.csv` differs from the hash recorded for the active model.

### 3.3 Evaluate page

Select a dataset split to inspect its predictions and evaluation results.

Test evaluation requires explicit confirmation. Test evaluations are logged in `.test_evaluations.json` under the dataset directory.

The dataset summary reads catalogue records without opening the photographs. Evaluation opens images from the selected split.

Do not use test results to tune the classifier, change the decision threshold, redraw splits or remove difficult but valid images.

### 3.4 Restart after changing model files

The server caches loaded pipelines by model ID.

If `pipeline.joblib` is replaced while the server is running, the application may continue using the previously loaded model. Stop and restart the server after replacing model files. Refreshing the browser alone does not reload the model from disk.

## 4. Project Structure

```
main.py
README.md
requirements.txt
requirements-lock.txt

src/
    app.py
    application/
    camera/
    data/
    preprocessing/
    features/
    training/
    experiments/
    storage/
    utils/

webui/
configs/
scripts/
tools/
tests/

models/
    final/
        pipeline.joblib
        pipeline_metadata.json

dataset_dev/
    images.csv
    locations.csv
    split_report.json
    images/
        inside/
        outside/
```

Generated analysis outputs are stored in directories selected with the relevant script's output argument. These directories are not necessarily tracked in Git.

## 5. Dataset

### 5.1 Dataset delivery

The photographs are not stored in Git.

Use the dataset supplied with the submission. If the dataset is distributed separately through cloud storage, obtain it using the link in `DATASET_LINK.txt` and place it under `dataset_dev/`.

A cloud dataset link must:

* Allow viewing and downloading without signing in.
* Require no password or access request.
* Provide the complete dataset.
* Remain accessible until grading is complete.

The same sharing link must appear in the report.

### 5.2 Expected layout

```
dataset_dev/
    images.csv
    locations.csv
    split_report.json
    images/
        inside/
            in_001/
                image_01.jpg
        outside/
            out_001/
                image_01.jpg
```

Keep the relative paths consistent with `images.csv`.

### 5.3 Dataset records

The core fields in `images.csv` are:


| Field           | Purpose                                      |
| --------------- | -------------------------------------------- |
| `image_id`      | Unique image identifier                      |
| `relative_path` | Image path relative to the dataset directory |
| `label`         | Numeric class: 1 for inside, 0 for outside   |
| `group_id`      | Location-group identifier                    |
| `split`         | `train`,`validation`or`test`                 |

The catalogue may also contain descriptive fields such as `session_id`, `scene_type` and `lighting`. These are dataset records, not classifier inputs.

The fields in `locations.csv` include:


| Field                  | Purpose                                |
| ---------------------- | -------------------------------------- |
| `group_id`             | Location-group identifier              |
| `class_name`           | `inside`or`outside`                    |
| `location_description` | Human-readable location description    |
| `boundary_verified`    | Record of manual boundary verification |

### 5.4 Frozen dataset counts


| Split      | Inside images | Inside groups | Outside images | Outside groups | Total images | Total groups |
| ---------- | ------------- | ------------- | -------------- | -------------- | ------------ | ------------ |
| Train      | 52            | 8             | 58             | 9              | 110          | 17           |
| Validation | 15            | 2             | 20             | 3              | 35           | 5            |
| Test       | 17            | 2             | 19             | 3              | 36           | 5            |
| Total      | 84            | 12            | 97             | 15             | 181          | 27           |

### 5.5 Group-based splitting

Splits are made by whole location group, separately within each class, aiming for approximately:

* 60% training groups.
* 20% validation groups.
* 20% test groups.

The target concerns location groups, not exact image-level proportions.

Nearby views of the same scene, repeat visits, burst frames and edited copies must remain in the same group and split.

The submitted split is frozen. Do not redraw it when reproducing the experiments.

### 5.6 Read-only validation

```
python scripts/validate_dataset.py --dataset-dir dataset_dev --require-split --require-verified
```

A verification flag records a manual check. Software cannot establish geographic correctness merely from the flag.

## 6. Final Model

The final model files are:

```
models/final/pipeline.joblib
models/final/pipeline_metadata.json
```

### 6.1 Pipeline

```
Image
  -> EXIF orientation correction
  -> RGB conversion
  -> 256 x 256 squash resize
  -> HSV + HOG + LBP feature extraction
  -> StandardScaler
  -> L1-regularised logistic regression
  -> Inside/outside prediction
```

### 6.2 Configuration


| Component              | Final setting       |
| ---------------------- | ------------------- |
| Image size             | 256 × 256          |
| Resize mode            | `squash`            |
| Colour features        | HSV histograms      |
| Colour bins            | 16 per channel      |
| Colour grid            | 1 × 1              |
| Colour dimensions      | 48                  |
| HOG orientations       | 9                   |
| HOG pixels per cell    | 32 × 32            |
| HOG cells per block    | 2 × 2              |
| HOG dimensions         | 1,764               |
| LBP method             | Uniform             |
| LBP points             | 8                   |
| LBP radius             | 1                   |
| LBP dimensions         | 10                  |
| Total input dimensions | 1,822               |
| Scaling                | `StandardScaler`    |
| Classifier             | Logistic regression |
| Regularisation         | L1                  |
| C                      | 0.1                 |
| Solver                 | `liblinear`         |
| Class weighting        | `balanced`          |
| Maximum iterations     | 10,000              |
| Recorded seed          | 42                  |

The saved model has 19 effective nonzero coefficients using an absolute-coefficient threshold of `1e-8`.

All 1,822 descriptors are still extracted. Having 19 active coefficients does not mean that only 19 descriptors are computed.

## 7. Recorded Results

These values summarise the completed experiments. The corresponding saved metric files and report should be retained as the experiment records.


| Evaluation                                      | Accuracy       | Macro-F1                       |
| ----------------------------------------------- | -------------- | ------------------------------ |
| Final model: training                           | 92.73%         | 0.9269                         |
| Final model: validation                         | 71.43%         | 0.7141                         |
| Full-feature training-group CV: mean pooled OOF | 76.06%         | 0.7547                         |
| Final model: held-out test                      | 97.22% (35/36) | Refer to exported test metrics |

The repeated training-group CV comparison uses three folds repeated with three split seeds.

CV models are fitted on subsets of the training set. Their scores are not directly interchangeable with scores from the final model trained on the complete training split.

### 7.1 Validation results by location


| Group     | True class | Correct | Accuracy |
| --------- | ---------- | ------- | -------- |
| `out_001` | outside    | 1/4     | 25.0%    |
| `out_014` | outside    | 3/8     | 37.5%    |
| `in_011`  | inside     | 6/8     | 75.0%    |
| `in_013`  | inside     | 7/7     | 100.0%   |
| `out_013` | outside    | 8/8     | 100.0%   |

`out_001` contains challenging nighttime scenes. Its performance does not, by itself, isolate the effect of lighting from other location-specific differences.

### 7.2 Test results by location


| Group     | True class | Correct | Accuracy |
| --------- | ---------- | ------- | -------- |
| `in_005`  | inside     | 10/11   | 90.9%    |
| `in_006`  | inside     | 6/6     | 100.0%   |
| `out_003` | outside    | 5/5     | 100.0%   |
| `out_005` | outside    | 11/11   | 100.0%   |
| `out_011` | outside    | 3/3     | 100.0%   |

Only five test location groups are available. Images from one location are correlated, so high image-level test accuracy should not be interpreted as reliable performance at arbitrary unseen locations.

The observation that test scenes appeared easier is qualitative, rather than based on a predefined difficulty measure.

## 8. Run the Tests

```
python main.py test
```

Alternatively:

```
python -m unittest discover -s tests -t . -v
```

Run these commands in the packaged environment before submission. The presence of test code does not itself establish that all tests pass in a fresh installation.

## 9. Reproduce the Hue #0 Analysis and Retraining Comparison

Retraining is optional for application use. Do not overwrite the submitted final pipeline during verification.

### 9.1 Full comparison

```
python tools/analyze_train_hue0.py \
  --mode compare \
  --pipeline models/final/pipeline.joblib \
  --dataset-dir dataset_dev \
  --output-dir results/reproduce_hue0_01
```

This command:

1. Loads the saved final pipeline as a configuration template.
2. Analyzes the saved model's train and validation predictions.
3. Exports active coefficients and per-image contributions.
4. Performs a fixed-model Hue #0 contribution-zeroing diagnostic.
5. Compares `full` and `drop_hue0` using training-only grouped CV.
6. Refits the scaler and classifier inside each fold.
7. Fits both candidates on the complete training split.
8. Exports train and validation metrics and candidate pipelines.

It does not open test image bytes or automatically replace the final model.

The output directory must not already exist.

### 9.2 Retraining scope

The `full` arm clones the saved pipeline's feature extractor, scaler and classifier settings.

This is template-based retraining. It does not rerun the entire historical model search from scratch.

No additional C search is performed.

### 9.3 Main outputs


| Output                           | Purpose                                      |
| -------------------------------- | -------------------------------------------- |
| `SUMMARY.md`                     | Overall comparison                           |
| `run_metadata.json`              | Configuration, environment and hashes        |
| `split_counts.csv`               | Class/split image and group counts           |
| `development_manifest.csv`       | Train/validation records and image hashes    |
| `saved_active_contributions.csv` | Active-feature contributions                 |
| `hue0_diagnostic.csv`            | Original and contribution-zeroed predictions |
| `hue0_fixed_broken.csv`          | Fixed/broken counts by group                 |
| `cv_summary.csv`                 | Mean repeated group-CV results               |
| `cv_paired_macro_f1.csv`         | Paired repeat-level differences              |
| `cv_feature_stability.csv`       | Selection frequency across folds             |
| `cv_warnings.json`               | Recorded fitting warnings                    |
| `cv_recommendation.json`         | Training-CV recommendation                   |
| `candidates/`                    | Experimental fitted pipelines                |

Candidate metadata is not guaranteed to be a drop-in replacement for the application's final-model metadata.

### 9.4 Recorded comparison


| Candidate                 | Training accuracy | Validation accuracy | Mean group-CV macro-F1 |
| ------------------------- | ----------------- | ------------------- | ---------------------- |
| Full features             | 92.73%            | 71.43%              | 0.7547                 |
| Remove Hue #0 and retrain | 96.36%            | 77.14%              | 0.6130                 |

The full-feature model was retained.

Although removing Hue #0 improved fixed validation accuracy, it reduced mean training-group CV performance.

The hypothesis was developed after inspecting validation errors. This comparison is exploratory, not independent confirmation of generalisation.

### 9.5 Analysis without candidate training

```
python tools/analyze_train_hue0.py \
  --mode analyze \
  --pipeline models/final/pipeline.joblib \
  --dataset-dir dataset_dev \
  --output-dir results/reproduce_analysis_01
```

Analysis-only mode does not produce a CV recommendation.

### 9.6 Interpretation of contribution zeroing

For a linear classifier, each contribution is:

```
contribution_j = coefficient_j × standardized_feature_j
```

Setting a standardized contribution to zero places the feature at its training mean. It does not:

* Set the original histogram bin to zero.
* Remove coloured pixels.
* Retrain other coefficients.

Removing the input feature and retraining is a separate experiment.

## 10. Reproduce the Learning Curve

```
python tools/learning_curve_groups.py \
  --pipeline models/final/pipeline.joblib \
  --dataset-dir dataset_dev \
  --output-dir results/reproduce_learning_curve_01 \
  --repeats 20
```

The script:

* Uses whole training location groups.
* Selects nested subsets separately within each class.
* Keeps validation fixed.
* Refits scaling and classification for each subset.
* Does not decode test images.

Outputs include:

```
runs.csv
summary.csv
protocol.json
learning_curve.png
learning_curve.pdf
```

The shaded bands represent ±1 standard deviation across training-group subsets. They are not confidence intervals.

At full training size, repeated subsets contain the same training groups, so a narrow or zero-width band does not establish uncertainty-free generalisation.

## 11. Inspect Active Features on an Image

```
python tools/inspect_active_features.py \
  --pipeline models/final/pipeline.joblib \
  --dataset-dir dataset_dev \
  --output-dir results/reproduce_active_features_01
```

Without `--image-id`, the script selects the first validation image.

To inspect a specific train or validation image, add its exact `image_id` from `images.csv`:

```
python tools/inspect_active_features.py \
  --pipeline models/final/pipeline.joblib \
  --dataset-dir dataset_dev \
  --image-id photo_2026_10_16_10_10_56_9_4dc43e036647 \
  --output-dir results/reproduce_ugarden_01
```

Outputs include:

```
active_features.csv
feature_locations.png
feature_locations.pdf
```

The figure shows:

* The preprocessed image.
* Active HOG cell locations.
* Per-feature contributions to the prediction.

HOG rectangles show feature-cell support, not pixel-level saliency or recognised objects. Global colour and LBP histograms do not identify a unique image region.

This script only accepts train and validation images. It is not the test-evaluation entry point.

## 12. Other Training Entry Points

```
python main.py train --help
```

```
python -m src.train --help
```

These are generic training/search interfaces. Their defaults should not be assumed to reproduce the final logistic-regression model.

In particular, a `center_crop` training example does not reproduce the final `squash` configuration.

Use the template-based comparison in Section 9 to refit the final settings without changing the submitted final pipeline.

### 12.1 Historical experiments

The repository also contains:

```
scripts/run_polyu_lr.py
scripts/prepare_polyu_feature_experiment.py
scripts/run_feature_experiment.py
```

Historical configurations may depend on earlier baseline artifacts or paths that are not included in a minimal submission.

For example, `configs/polyu_lr_02.json` references `models_runs/` and `results_runs/`, and enables `allow_unverified`. It is a historical development configuration, not the final submission recipe.

Inspect configurations and supported dry-run options before using these entry points.

## 13. Dataset Preparation Utilities

These utilities are for dataset development, not routine reproduction of the frozen submission:


| Script                         | Purpose                                                |
| ------------------------------ | ------------------------------------------------------ |
| `scripts/update_images_csv.py` | Scan images and update catalogue records               |
| `scripts/split_dataset.py`     | Generate location-group splits                         |
| `scripts/validate_dataset.py`  | Check catalogue consistency                            |
| `tools/organise_photos.py`     | Prepare a collection manifest and copy new photographs |

Do not force a new split or modify the submitted dataset after reviewing test results.

If a genuine data error is discovered, document the correction and its consequences rather than silently changing the evaluation data.

## 14. Diagnostic and Maintenance Tools


| Tool                         | Purpose                                                    |
| ---------------------------- | ---------------------------------------------------------- |
| `analyze_train_hue0.py`      | Contribution analysis, ablation, grouped CV and retraining |
| `learning_curve_groups.py`   | Whole-location learning curve                              |
| `inspect_active_features.py` | Active-feature CSV and visual explanation                  |
| `find_duplicates.py`         | Near-duplicate diagnostics                                 |
| `audit_sources.py`           | Resolution, compression and sharpness diagnostics          |
| `nearest_train.py`           | Feature-space comparison with training images              |
| `preview_modes.py`           | Resize-mode inspection                                     |
| `make_variants.py`           | Generate diagnostic image variants                         |
| `summarise_variants.py`      | Summarise diagnostic variant results                       |
| `organise_photos.py`         | Optional dataset-development utility                       |
| `consolidate_tests.py`       | Historical repository-maintenance utility                  |

`consolidate_tests.py` is not a test runner. It was designed to move test packages and rewrite references.

Diagnostic image variants are not independent additional dataset samples.

## 15. Known Limitations

* The dataset contains a limited number of location groups.
* Performance varies substantially across locations.
* Some outside scenes share visual statistics with campus scenes.
* Nighttime conditions may differ from the training coverage.
* Global colour features can dominate a decision without identifying semantic objects.
* HOG features depend on local gradient structure and spatial layout.
* Global LBP features summarise texture without identifying its location.
* Resizing changes image geometry and may affect descriptors.
* The high test accuracy applies to the evaluated test groups, not all possible locations.
* A prediction is not proof of a photograph's geographic location.

The Hue #0 analysis shows that one feature can help some groups while harming others. A large coefficient or a single failed example does not, by itself, prove that a feature should be removed.

## 16. Troubleshooting

### No model found

Confirm that this file exists:

```
models/final/pipeline.joblib
```

Launch with:

```
python -m src.app --outputs-dir models/final --dataset-dir dataset_dev
```

### Predictions differ after replacing a model

1. Stop the server.
2. Confirm the intended model file.
3. Restart the server.
4. Check the active model.
5. Compare the exact image and model hashes if necessary.

### Evaluation cannot find images

Obtain the complete dataset and preserve the relative paths in `images.csv`.

### Output directory already exists

Choose a new output directory. Preserve previous experiment outputs instead of overwriting them.

### Boundary verification fails

Inspect the actual location and update its record only after verification. Do not bypass verification merely to produce a final result.

### Saved model cannot be loaded

Check:

* Python version.
* Package versions.
* Whether commands are run from the project root.
* Whether `src/` is importable.
* Whether the model file is complete and trusted.

### Convergence warnings appear

Inspect the saved diagnostics. Do not silently treat a candidate with unresolved convergence problems as validated.

## 17. Reproducibility and Submission Checklist

Before submission:

* Confirm that the final model and metadata correspond.
* Preserve the frozen dataset records.
* Include overall and per-class test metrics.
* Include representative correct and incorrect predictions.
* Retain the analysis scripts and outputs used in the report.
* Run the tests.
* Extract the intended submission ZIP into a fresh folder.
* Install dependencies and launch the application using these instructions.
* Confirm that no local absolute paths are required.
* Confirm that the dataset is included or its sharing link works while signed out.
* Use the required report and code ZIP filenames.
* Keep credentials, virtual environments, caches and Git internals out of the ZIP.

The report PDF is submitted separately from the implementation ZIP.

## 18. Generative AI Assistance

Generative AI assisted implementation, debugging and analysis tooling.

The report should explain:

* Which parts received AI assistance.
* What was checked by the student.
* Which mistakes or limitations were observed.
* How corrections and model-selection decisions were made.

Examples from this project include:

* Investigating disagreement between application output and direct pipeline inference.
* Recognising the effect of cached model state.
* Distinguishing a fixed-model contribution-zeroing diagnostic from feature removal and retraining.
* Retaining the original model despite an alternative's higher validation accuracy.

Do not claim a clean-environment run, complete geographic audit or other verification until it has actually been completed.
