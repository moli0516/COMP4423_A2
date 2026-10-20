# PolyU Main Campus Scene Dataset

# PolyU Main Campus Scene Dataset

Dataset documentation for COMP4423 Computer Vision, Assignment 2 (2026/27 Semester 1).

## 1. Current status

This is a self-collected dataset under construction. It is not yet a final, split, or evaluated dataset.

- Five initial candidate photographs have been reviewed in conversation.
- Their provisional location group is `in_001`: the courtyard area around Jockey Club Auditorium and Lam Tai Fai Amphitheatre.
- Exact camera positions and geographic labels must be checked against the official campus boundary before acceptance.
- Photographs containing prominent or identifiable people should be replaced with suitable retakes before inclusion.
- The number of accepted images is not yet confirmed.
- No final training, validation, or test split has been assigned.
- Empty numbered folders created by the scaffold are placeholders, not collected location groups.

Update this README and both CSV records as collection and review progress. Do not present provisional information as verified results.

## 2. Task and class labels

The task is geographic scene classification, not indoor/outdoor classification.


| Class name | Numeric label | Definition                                                                          |
| ---------- | ------------: | ----------------------------------------------------------------------------------- |
| `inside`   |           `1` | The camera position and the main depicted scene are both within PolyU main campus.  |
| `outside`  |           `0` | The camera position and the main depicted scene are both outside PolyU main campus. |

Only the main campus is the positive class. PolyU-affiliated off-campus premises remain `outside`.

Use PolyU's official campus map to establish geographic ground truth. Do not infer labels from red bricks, logos, signs, or architectural appearance alone.

Exclude uncertain boundary positions, gate thresholds, connecting bridges, and views whose main scene looks across the campus boundary.

## 3. Collection scope

The initial collection focuses on outdoor scenes:

- Buildings and their surroundings.
- Courtyards and open spaces.
- Pedestrian paths and permitted covered walkways.
- Greenery and landscaped spaces.

Both classes should contain comparable scene types. Include some outside scenes that resemble campus scenes rather than collecting only easy negatives.

Indoor images are not required for the initial collection. If added, record the actual scope and preserve geographic labels and collection permissions. A dataset primarily containing outdoor scenes does not establish model reliability for indoor scenes.

All included photographs must be taken by the student. Public photographs, screenshots, and photographs of other images are not substitutes for the required self-collected dataset.

No fixed image count or location-group count is specified by the assignment. Final scale and diversity must be justified using the actual collected data.

## 4. Directory structure

Paths below are relative to the project root:

```text
assignment_2/
├── originals/
│   └── session_001/          # Original camera files; not loaded for training
├── review/
│   └── session_001/          # Candidates awaiting review; not loaded for training
└── dataset/
    ├── images/
    │   ├── inside/
    │   │   ├── in_001/
    │   │   └── ...
    │   └── outside/
    │       ├── out_001/
    │       └── ...
    ├── images.csv
    ├── locations.csv
    └── dataset_readme.md
```

Store one authoritative working copy of each accepted image under `dataset/images/`. Keep originals separately and do not recursively load images from the entire project.

The working folders may contain fewer groups than the scaffold creates. Unused empty folders may be removed; they must not be counted in dataset statistics.

## 5. Location groups: the authoritative grouping rule

A location group represents a physical scene, building, courtyard, or landmark. It is not a category of visual appearance or a collection condition.

- Different views of the same courtyard, building, or landmark stay together.
- Repeat visits, burst/video frames, and edited copies stay with the original scene group.
- Different locations do not become one group merely because both are sunny, brick-built, or landscaped.
- Different scene types within one courtyard do not automatically become different location groups.
- Inspect overlap between candidate groups. Merge groups when they represent repeated views of the same physical scene; document the decision.
- An incidental distant background is not automatically a reason to merge every image across the campus. Review whether the main scene or landmark is being repeated.

The previous 25-pair scene checklist is a diversity checklist, not a list of 25 confirmed groups. Candidate codes such as `IN-A` and `OUT-A` are planning references, not authoritative dataset IDs.

Use only `in_001`, `in_002`, etc. and `out_001`, `out_002`, etc. as formal group IDs. IDs need not remain consecutive after merges. Update paths and CSV records together when changing a group.

## 6. Metadata files

Save CSV files as UTF-8 with one header row. Quote any value containing commas, line breaks, or quotation marks using standard CSV quoting.

### 6.1 `locations.csv`: one row per documented location group

```csv
group_id,class_name,location_description,boundary_verified
```


| Field                  | Meaning                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| `group_id`             | Formal location-group ID matching the working folder.                                            |
| `class_name`           | `inside` or `outside`.                                                                           |
| `location_description` | Actual location and useful camera-position notes; do not use only a broad label such as "PolyU". |
| `boundary_verified`    | `yes` after verification; `pending` while awaiting verification.                                 |

Candidates may be documented as `pending`, but their images must not enter the accepted image manifest until the label and collection conditions are verified.

### 6.2 `images.csv`: one row per accepted image

```csv
image_id,relative_path,label,group_id,session_id,scene_type,lighting,split
```


| Field           | Meaning                                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------------------------------- |
| `image_id`      | Unique image identifier.                                                                                             |
| `relative_path` | Image path relative to`dataset/`, using forward slashes; for example, `images/inside/in_001/<actual_filename>.jpeg`. |
| `label`         | Integer`1` for inside or `0` for outside.                                                                            |
| `group_id`      | Group matching`locations.csv` and the image folder.                                                                  |
| `session_id`    | Collection session; a repeat visit can have a new session ID without changing group ID.                              |
| `scene_type`    | Descriptive tags such as`building;courtyard;walkway`. These do not define grouping.                                  |
| `lighting`      | Observed condition such as`sunny`, `overcast`, or `shade`; use `not_recorded` if unknown.                            |
| `split`         | Blank during collection; later`train`, `validation`, or `test`.                                                      |

Do not add placeholder images or hypothetical rows. Keep image IDs and paths unique. If different devices reuse a filename, choose a unique working filename and record the change without overwriting existing data.

## 7. Collection and acceptance workflow

1. Record the actual location and provisional group.
2. Verify both camera position and main scene against the main-campus boundary.
3. Photograph only where permitted; avoid identifiable people, personal information, and restricted areas.
4. Capture useful surroundings, not isolated objects or close-up signs alone.
5. Vary viewpoint, distance, lighting, and background where practical.
6. Back up original files.
7. Inspect full-resolution images and group contact sheets.
8. Reject or retake ambiguous, privacy-sensitive, unreadable, severely blurred, or excessively redundant candidates. Record reasons.
9. Copy accepted working images into the formal group folder.
10. Update both CSV files.

A practical starting target is around 10–15 useful views per location, not an official requirement. Prioritise new genuine locations over large numbers of near-identical images.

Do not exclude valid images merely because the classifier finds them difficult. Do not remove difficult test images after viewing model performance.

## 8. Group-level split protocol

The assignment requires splitting location groups separately within each class, aiming for 60% training, 20% validation, and 20% test, with both classes in every split.

1. Review and finalise physical location groups.
2. List unique accepted groups separately for inside and outside.
3. Use a documented fixed random seed to allocate groups.
4. Assign every image in a group to that group's single split.
5. Save and freeze `images.csv` before model comparison.
6. Report actual image counts and group counts for every class and split.

Image proportions may differ from group proportions if group sizes vary. State the actual counts rather than assuming exact percentages.

Training is used to fit models and learned transformations. Validation is used to select features and hyperparameters. Test is reserved for final evaluation, not model tuning or repeated split selection.

Pilot images used to adjust models or preprocessing should remain in training or validation rather than later being presented as unseen test data.

### Augmentation

If used, apply augmentation only to training images after the split. Keep derivatives associated with their parent image and group. Report original counts separately from augmented training counts. No augmentation has been documented yet.

## 9. Image handling and permitted model inputs

Working formats may be `.jpg`, `.jpeg`, or `.png`, provided the implemented loader supports them. The brief does not prescribe a format or resolution.

If converting HEIC or another camera format, use a consistent conversion and verify image orientation, readability, and file correspondence. A converted copy is not a new independent sample.

Implement orientation handling, colour conversion, resizing, and feature extraction in shared code. Document the actual parameters used; they are not yet finalised. Preserve the same processing in training, evaluation, and the application.

For the model:

- `X`: pixel-derived visual features only.
- `y`: geographic class label.
- Do not use paths, filenames, group IDs, split names, scene tags, lighting records, GPS/EXIF location, or external lookups as predictive features.

The assignment requires a traditional classifier and excludes deep learning. The deployment application uses a saved fitted pipeline, not location metadata or a dataset lookup.

## 10. Validation checklist

Before training, check that:

- Every manifest path exists and each image can be decoded.
- Image IDs and relative paths are unique.
- Labels are only `0` or `1` and agree with the group class.
- Every referenced group exists in `locations.csv` and is boundary-verified.
- Each group has exactly one class and, after splitting, exactly one split.
- Each split contains both classes.
- Exact duplicates do not cross splits.
- Contact sheets do not reveal the same physical scene hidden under different group IDs.
- Only accepted images are in the manifest.
- Image/group summary counts are calculated from the manifest, not empty folders.

## 11. Dataset summary: pending completion

Replace `Pending` with measured counts after final review and splitting. Empty scaffold folders are not groups.


| Class   | Train images / groups | Validation images / groups | Test images / groups | Total images / groups |
| ------- | --------------------- | -------------------------- | -------------------- | --------------------- |
| Inside  | Pending               | Pending                    | Pending              | Pending               |
| Outside | Pending               | Pending                    | Pending              | Pending               |
| Total   | Pending               | Pending                    | Pending              | Pending               |

Record before final submission:

- Collection date range: Pending
- Devices used: Pending
- Final accepted image count: Pending
- Final accepted location-group count: Pending
- Split seed and split-generation method: Pending
- Dataset version and freeze date: Pending
- Augmentation policy: Pending
- Preprocessing configuration: Pending

## 12. Limitations and reporting

Discuss actual limitations supported by the collected data: limited location coverage, outdoor-focused scope, lighting or viewpoint coverage, repeated backgrounds, and campus-like outside scenes.

Relate observed prediction failures to feature-extraction limitations in the report. Do not claim balanced coverage, verified counts, high accuracy, or reproducibility until checked.

Document any GenAI assistance, actual errors observed, verification performed, and the student's corrections. This README does not replace the required report or its template questions.

## 13. Submission and access

Submit the complete dataset in the implementation ZIP, or use the permitted cloud-drive option.

For cloud submission:

- Put the sharing link in `DATASET_LINK.txt` inside the implementation ZIP and in the report.
- Allow complete viewing and downloading without sign-in, password, access requests, or special permissions.
- Test access in a signed-out/private browser.
- Keep the link accessible until grading is complete.

Include the curated images, CSV records, and this README. Separate backups and rejected candidates are not the accepted dataset and should not be loaded by reproduction scripts. Do not publish privacy-sensitive rejected originals through a public dataset link.

## 14. Requirement source

The authoritative specification is `Assignment-2-requirements-COMP4423_202627S1.pdf`, especially the class definitions, Tasks 1–2, evaluation/application requirements, and submission instructions. The official campus map is the geographic reference.

The instructor's written clarification takes precedence over suggested collection targets or folder conventions in this README.

Dataset documentation for COMP4423 Computer Vision, Assignment 2 (2026/27 Semester 1).

## 1. Current status

This is a self-collected dataset under construction. It is not yet a final, split, or evaluated dataset.

* Five initial candidate photographs have been reviewed in conversation.
* Their provisional location group is `in_001`: the courtyard area around Jockey Club Auditorium and Lam Tai Fai Amphitheatre.
* Exact camera positions and geographic labels must be checked against the official campus boundary before acceptance.
* Photographs containing prominent or identifiable people should be replaced with suitable retakes before inclusion.
* The number of accepted images is not yet confirmed.
* No final training, validation, or test split has been assigned.
* Empty numbered folders created by the scaffold are placeholders, not collected location groups.

Update this README and both CSV records as collection and review progress. Do not present provisional information as verified results.

## 2. Task and class labels

The task is geographic scene classification, not indoor/outdoor classification.


| Class name | Numeric label | Definition                                                                          |
| ---------- | ------------- | ----------------------------------------------------------------------------------- |
| `inside`   | `1`           | The camera position and the main depicted scene are both within PolyU main campus.  |
| `outside`  | `0`           | The camera position and the main depicted scene are both outside PolyU main campus. |

Only the main campus is the positive class. PolyU-affiliated off-campus premises remain `outside`.

Use PolyU's official campus map to establish geographic ground truth. Do not infer labels from red bricks, logos, signs, or architectural appearance alone.

Exclude uncertain boundary positions, gate thresholds, connecting bridges, and views whose main scene looks across the campus boundary.

## 3. Collection scope

The initial collection focuses on outdoor scenes:

* Buildings and their surroundings.
* Courtyards and open spaces.
* Pedestrian paths and permitted covered walkways.
* Greenery and landscaped spaces.

Both classes should contain comparable scene types. Include some outside scenes that resemble campus scenes rather than collecting only easy negatives.

Indoor images are not required for the initial collection. If added, record the actual scope and preserve geographic labels and collection permissions. A dataset primarily containing outdoor scenes does not establish model reliability for indoor scenes.

All included photographs must be taken by the student. Public photographs, screenshots, and photographs of other images are not substitutes for the required self-collected dataset.

No fixed image count or location-group count is specified by the assignment. Final scale and diversity must be justified using the actual collected data.

## 4. Directory structure

Paths below are relative to the project root:

```plaintext
│   └── session_001/          # Original camera files; not loaded for training
├── review/
│   └── session_001/          # Candidates awaiting review; not loaded for training
└── dataset/
├── images/
│   ├── inside/
│   │   ├── in_001/
│   │   └── ...
│   └── outside/
│       ├── out_001/
│       └── ...
├── images.csv
├── locations.csv
└── dataset_readme.md
```

Store one authoritative working copy of each accepted image under `dataset/images/`. Keep originals separately and do not recursively load images from the entire project.

The working folders may contain fewer groups than the scaffold creates. Unused empty folders may be removed; they must not be counted in dataset statistics.

## 5. Location groups: the authoritative grouping rule

A location group represents a physical scene, building, courtyard, or landmark. It is not a category of visual appearance or a collection condition.

* Different views of the same courtyard, building, or landmark stay together.
* Repeat visits, burst/video frames, and edited copies stay with the original scene group.
* Different locations do not become one group merely because both are sunny, brick-built, or landscaped.
* Different scene types within one courtyard do not automatically become different location groups.
* Inspect overlap between candidate groups. Merge groups when they represent repeated views of the same physical scene; document the decision.
* An incidental distant background is not automatically a reason to merge every image across the campus. Review whether the main scene or landmark is being repeated.

The previous 25-pair scene checklist is a diversity checklist, not a list of 25 confirmed groups. Candidate codes such as `IN-A` and `OUT-A` are planning references, not authoritative dataset IDs.

Use only `in_001`, `in_002`, etc. and `out_001`, `out_002`, etc. as formal group IDs. IDs need not remain consecutive after merges. Update paths and CSV records together when changing a group.

## 6. Metadata files

Save CSV files as UTF-8 with one header row. Quote any value containing commas, line breaks, or quotation marks using standard CSV quoting.

## 6.1 `locations.csv`: one row per documented location group

```plaintext
group_id,class_name,location_description,boundary_verified
```


| Field                  | Meaning                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| `group_id`             | Formal location-group ID matching the working folder.                                            |
| `class_name`           | `inside`or`outside`.                                                                             |
| `location_description` | Actual location and useful camera-position notes; do not use only a broad label such as "PolyU". |
| `boundary_verified`    | `yes`after verification;`pending`while awaiting verification.                                    |

Candidates may be documented as `pending`, but their images must not enter the accepted image manifest until the label and collection conditions are verified.

## 6.2 `images.csv`: one row per accepted image

```plaintext
image_id,relative_path,label,group_id,session_id,scene_type,lighting,split
```


| Field           | Meaning                                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------- |
| `image_id`      | Unique image identifier.                                                                                            |
| `relative_path` | Image path relative to`dataset/`, using forward slashes; for example,`images/inside/in_001/<actual_filename>.jpeg`. |
| `label`         | Integer`1`for inside or`0`for outside.                                                                              |
| `group_id`      | Group matching`locations.csv`and the image folder.                                                                  |
| `session_id`    | Collection session; a repeat visit can have a new session ID without changing group ID.                             |
| `scene_type`    | Descriptive tags such as`building;courtyard;walkway`. These do not define grouping.                                 |
| `lighting`      | Observed condition such as`sunny`,`overcast`, or`shade`; use`not_recorded`if unknown.                               |
| `split`         | Blank during collection; later`train`,`validation`, or`test`.                                                       |

Do not add placeholder images or hypothetical rows. Keep image IDs and paths unique. If different devices reuse a filename, choose a unique working filename and record the change without overwriting existing data.

## 7. Collection and acceptance workflow

1. Record the actual location and provisional group.
2. Verify both camera position and main scene against the main-campus boundary.
3. Photograph only where permitted; avoid identifiable people, personal information, and restricted areas.
4. Capture useful surroundings, not isolated objects or close-up signs alone.
5. Vary viewpoint, distance, lighting, and background where practical.
6. Back up original files.
7. Inspect full-resolution images and group contact sheets.
8. Reject or retake ambiguous, privacy-sensitive, unreadable, severely blurred, or excessively redundant candidates. Record reasons.
9. Copy accepted working images into the formal group folder.
10. Update both CSV files.

A practical starting target is around 10–15 useful views per location, not an official requirement. Prioritise new genuine locations over large numbers of near-identical images.

Do not exclude valid images merely because the classifier finds them difficult. Do not remove difficult test images after viewing model performance.

## 8. Group-level split protocol

The assignment requires splitting location groups separately within each class, aiming for 60% training, 20% validation, and 20% test, with both classes in every split.

1. Review and finalise physical location groups.
2. List unique accepted groups separately for inside and outside.
3. Use a documented fixed random seed to allocate groups.
4. Assign every image in a group to that group's single split.
5. Save and freeze `images.csv` before model comparison.
6. Report actual image counts and group counts for every class and split.

Image proportions may differ from group proportions if group sizes vary. State the actual counts rather than assuming exact percentages.

Training is used to fit models and learned transformations. Validation is used to select features and hyperparameters. Test is reserved for final evaluation, not model tuning or repeated split selection.

Pilot images used to adjust models or preprocessing should remain in training or validation rather than later being presented as unseen test data.

## Augmentation

If used, apply augmentation only to training images after the split. Keep derivatives associated with their parent image and group. Report original counts separately from augmented training counts. No augmentation has been documented yet.

## 9. Image handling and permitted model inputs

Working formats may be `.jpg`, `.jpeg`, or `.png`, provided the implemented loader supports them. The brief does not prescribe a format or resolution.

If converting HEIC or another camera format, use a consistent conversion and verify image orientation, readability, and file correspondence. A converted copy is not a new independent sample.

Implement orientation handling, colour conversion, resizing, and feature extraction in shared code. Document the actual parameters used; they are not yet finalised. Preserve the same processing in training, evaluation, and the application.

For the model:

* `X`: pixel-derived visual features only.
* `y`: geographic class label.
* Do not use paths, filenames, group IDs, split names, scene tags, lighting records, GPS/EXIF location, or external lookups as predictive features.

The assignment requires a traditional classifier and excludes deep learning. The deployment application uses a saved fitted pipeline, not location metadata or a dataset lookup.

## 10. Validation checklist

Before training, check that:

* Every manifest path exists and each image can be decoded.
* Image IDs and relative paths are unique.
* Labels are only `0` or `1` and agree with the group class.
* Every referenced group exists in `locations.csv` and is boundary-verified.
* Each group has exactly one class and, after splitting, exactly one split.
* Each split contains both classes.
* Exact duplicates do not cross splits.
* Contact sheets do not reveal the same physical scene hidden under different group IDs.
* Only accepted images are in the manifest.
* Image/group summary counts are calculated from the manifest, not empty folders.

## 11. Dataset summary: pending completion

Replace `Pending` with measured counts after final review and splitting. Empty scaffold folders are not groups.


| Class   | Train images / groups | Validation images / groups | Test images / groups | Total images / groups |
| ------- | --------------------- | -------------------------- | -------------------- | --------------------- |
| Inside  | Pending               | Pending                    | Pending              | Pending               |
| Outside | Pending               | Pending                    | Pending              | Pending               |
| Total   | Pending               | Pending                    | Pending              | Pending               |

Record before final submission:

* Collection date range: Pending
* Devices used: Pending
* Final accepted image count: Pending
* Final accepted location-group count: Pending
* Split seed and split-generation method: Pending
* Dataset version and freeze date: Pending
* Augmentation policy: Pending
* Preprocessing configuration: Pending

## 12. Limitations and reporting

Discuss actual limitations supported by the collected data: limited location coverage, outdoor-focused scope, lighting or viewpoint coverage, repeated backgrounds, and campus-like outside scenes.

Relate observed prediction failures to feature-extraction limitations in the report. Do not claim balanced coverage, verified counts, high accuracy, or reproducibility until checked.

Document any GenAI assistance, actual errors observed, verification performed, and the student's corrections. This README does not replace the required report or its template questions.

## 13. Submission and access

Submit the complete dataset in the implementation ZIP, or use the permitted cloud-drive option.

For cloud submission:

* Put the sharing link in `DATASET_LINK.txt` inside the implementation ZIP and in the report.
* Allow complete viewing and downloading without sign-in, password, access requests, or special permissions.
* Test access in a signed-out/private browser.
* Keep the link accessible until grading is complete.

Include the curated images, CSV records, and this README. Separate backups and rejected candidates are not the accepted dataset and should not be loaded by reproduction scripts. Do not publish privacy-sensitive rejected originals through a public dataset link.

## 14. Requirement source

The authoritative specification is `Assignment-2-requirements-COMP4423_202627S1.pdf`, especially the class definitions, Tasks 1–2, evaluation/application requirements, and submission instructions. The official campus map is the geographic reference.

The instructor's written clarification takes precedence over suggested collection targets or folder conventions in this README.
