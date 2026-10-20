# PolyU Main Campus Image Classification

COMP4423 Computer Vision — Assignment 2, 2026/27 Semester 1

A local, traditional-machine-learning application that classifies scene photographs as:

* `inside (1)`: taken within the PolyU main campus and depicting a scene within it.
* `outside (0)`: taken outside the main campus and depicting a scene outside it.

Labels describe geography, not indoors/outdoors or whether a picture looks like PolyU. Ground truth is checked against the official campus map. Uncertain boundary locations, gate thresholds, connecting bridges and cross-boundary views are excluded under the assignment rules; verify the actual location records rather than inferring compliance from a group name.

The classifier uses image pixels only. Filenames, folder labels and GPS/location metadata are not classifier features. EXIF orientation is applied only to display/decode the image correctly.

## 1. Quick start: run the submitted model

Run all commands from the project root containing `main.py` and `src/`.

## Environment

Use the Python/package environment recorded with the saved model whenever possible. The final model metadata records Python 3.13.7; the existing project documentation targets Python 3.10+, but compatibility with every older interpreter/package combination is not guaranteed.

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python -m venv .venv</span></code></span></div></figure></pre>

Activate it on macOS/Linux:

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span class="token token">source</span><span> .venv/bin/activate</span></code></span></div></figure></pre>

Or on Windows PowerShell:

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>powershell</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span class="token token punctuation">.</span><span>venv\Scripts\Activate</span><span class="token token punctuation">.</span><span>ps1</span></code></span></div></figure></pre>

Install the recorded dependencies:

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python -m pip </span><span class="token token">install</span><span> -r requirements-lock.txt</span></code></span></div></figure></pre>

For a fresh environment using unpinned package requirements instead:

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python -m pip </span><span class="token token">install</span><span> -r requirements.txt</span></code></span></div></figure></pre>

The unpinned option may install different versions and is not the preferred route for reproducing a serialized model. Only load trusted `joblib` files: deserialization can execute code.

## Launch

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python -m src.app --outputs-dir models/final --dataset-dir dataset_dev</span></code></span></div></figure></pre>

Open `http://127.0.0.1:8000`. Stop the server with Ctrl+C.

The narrow `models/final` search directory avoids exposing experimental pipelines as alternative final models. Application launch does not retrain the classifier or automatically evaluate the test split.

If port 8000 is occupied:

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python -m src.app --outputs-dir models/final --dataset-dir dataset_dev --port </span><span class="token token">8080</span><span> --no-browser</span></code></span></div></figure></pre>

Then open `http://127.0.0.1:8080`.

## 2. Application usage

## Predict

Upload/select an image. The application displays the image and returns:

* `Inside PolyU main campus`, or
* `Outside PolyU main campus`.

The predicted label comes from the saved classifier. For the final binary model, a positive decision score supports inside and a negative score supports outside. A decision score is not a probability.

The server processes uploaded image bytes through the saved preprocessing, feature extraction, scaler and classifier steps. No class label is requested as prediction input. The server rejects request bodies larger than 25 MiB.

The interface also provides live-camera functionality; camera input is optional, not required for normal image prediction. Browser camera permission is necessary.

## Config

Inspect the active pipeline, parameters, metadata and dataset summary. A mismatch warning may appear if the current `images.csv` differs from the hash recorded at training time.

## Evaluate

Choose train, validation or test to inspect overall and location-group results. Test evaluation requires explicit confirmation and is logged in `.test_evaluations.json` under the dataset directory.

Evaluation opens the selected split's images; simply opening the dataset/config summary does not.

Preserve the original evaluation history. Re-running the same frozen model for verification is not a new independent experiment. Do not use test results to tune models, delete difficult valid images or redraw splits.

## Important: restart after replacing model files

The application caches loaded pipelines by model ID. Replacing `pipeline.joblib` on disk does not automatically invalidate an already loaded object. Fully stop and restart the server after replacing a model; refreshing the browser alone is insufficient.

## 3. Dataset delivery and layout

Photographs are not stored in Git. Obtain the complete submitted dataset from the accompanying dataset archive or the sharing link in `DATASET_LINK.txt` included with the submission. Do not assume a Git clone contains the photographs.

For cloud delivery, the link must permit viewing/downloading without login, password or access requests. The same link must appear in the report. Confirm the link in a private/signed-out browser before submission.

Expected layout:

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>text</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>dataset_dev/
</span>  images.csv
  locations.csv
  split_report.json
  images/
    inside/
      in_001/
        ...
    outside/
      out_001/
        ...</code></span></div></figure></pre>

Core image fields are `image_id`, `relative_path`, `label`, `group_id`, `split`. The catalogue may also contain descriptive fields such as `session_id`, `scene_type` and `lighting`.

Location fields include `group_id`, `class_name`, `location_description`, `boundary_verified`.

## Frozen dataset counts


| Split      | Inside images | Inside groups | Outside images | Outside groups | Total images | Total groups |
| ---------- | ------------- | ------------- | -------------- | -------------- | ------------ | ------------ |
| Train      | 52            | 8             | 58             | 9              | 110          | 17           |
| Validation | 15            | 2             | 20             | 3              | 35           | 5            |
| Test       | 17            | 2             | 19             | 3              | 36           | 5            |
| Total      | 84            | 12            | 97             | 15             | 181          | 27           |

Splits are by whole location group, separately within each class. Nearby views of the same scene, repeat visits and edited copies must remain in one group/split. The target is approximately 60%/20%/20% of groups, not exact image-level proportions.

Read-only validation:

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python scripts/validate_dataset.py --dataset-dir dataset_dev --require-split --require-verified</span></code></span></div></figure></pre>

Do not edit the submitted records or rerun the splitter merely to reproduce results. A verification flag documents a manual boundary check; software cannot establish geographic truth from that flag alone.

## 4. Final model

The submitted model is `models/final/pipeline.joblib`, with configuration in `models/final/pipeline_metadata.json`.


| Component                     | Final setting                                                            |
| ----------------------------- | ------------------------------------------------------------------------ |
| Preprocessing                 | EXIF orientation, RGB conversion, 256 x 256 squash resize                |
| Colour                        | HSV histograms, 16 bins/channel, 1 x 1 grid; 48 dimensions               |
| HOG                           | 9 orientations, 32 x 32 pixels/cell, 2 x 2 cells/block; 1,764 dimensions |
| LBP                           | Uniform LBP, 8 points, radius 1; 10 dimensions                           |
| Total input dimensions        | 1,822                                                                    |
| Scaling                       | StandardScaler                                                           |
| Classifier                    | Logistic regression                                                      |
| Regularisation                | L1, C=0.1                                                                |
| Solver                        | liblinear                                                                |
| Class weighting               | balanced                                                                 |
| Maximum iterations            | 10,000                                                                   |
| Recorded seed                 | 42                                                                       |
| Effective active coefficients | 19 using absolute coefficient > 1e-8                                     |

All 1,822 descriptors are still extracted; 19 active coefficients do not imply that only 19 descriptors are computed.

## 5. Recorded results

These are the results recorded during the completed development/evaluation process. Use the final exported metric files and report as the authoritative experiment records; this README is a summary, not a new execution log.


| Evaluation                                      | Accuracy       | Macro-F1                  |
| ----------------------------------------------- | -------------- | ------------------------- |
| Final model: training                           | 92.73%         | 0.9269                    |
| Final model: validation                         | 71.43%         | 0.7141                    |
| Full-feature training-group CV: mean pooled OOF | 76.06%         | 0.7547                    |
| Final model: held-out test                      | 97.22% (35/36) | See exported test metrics |

The repeated CV comparison uses three group folds over three predeclared split seeds. Its fold models train on fewer images than the final model. CV, validation and test are different evaluations, not interchangeable estimates or confidence bounds.

## Location-dependent limitations

* Validation `out_001`: 1/4 correct; a challenging nighttime location.
* Validation `out_014`: 3/8 correct; some errors have strong positive Hue-bin-0 contributions.
* Validation `in_011` (U Garden): 6/8 correct.
* Test `in_005`: 10/11 correct; the remaining test groups are entirely correct.

Only five test location groups were available. Within-location photographs are correlated. High test accuracy does not establish reliable performance at arbitrary locations or under all lighting conditions. The observation that test scenes appeared easier is qualitative, not a controlled difficulty measurement. Nighttime performance at one location cannot isolate lighting from location-specific appearance.

## 6. Tests

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python main.py </span><span class="token token">test</span></code></span></div></figure></pre>

Alternative:

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python -m unittest discover -s tests -t </span><span class="token token">.</span><span> -v</span></code></span></div></figure></pre>

Run the tests in the packaged environment before submission. Their presence in Git does not itself prove they pass in a fresh installation.

## 7. Reproduce the final-setting comparison

The submitted final pipeline is the default for application use; retraining is optional. Do not overwrite it during verification.

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python tools/analyze_train_hue0.py </span><span class="token token punctuation">\</span><span>
</span><span>  --mode compare </span><span class="token token punctuation">\</span><span>
</span><span>  --pipeline models/final/pipeline.joblib </span><span class="token token punctuation">\</span><span>
</span><span>  --dataset-dir dataset_dev </span><span class="token token punctuation">\</span><span>
</span>  --output-dir results/reproduce_hue0_01</code></span></div></figure></pre>

This command:

1. Loads the final pipeline as a configuration template.
2. Analyzes its train/validation predictions and active contributions.
3. Compares `full` with `drop_hue0` using training-only grouped CV.
4. Refits scaling and classifier inside every fold.
5. Fits both candidates on the complete training split.
6. Evaluates train/validation and saves experimental candidate pipelines.
7. Does not read test image bytes or promote a candidate to final.

The `full` candidate reproduces the saved model's training configuration by cloning its transformer/scaler/classifier settings. This is template-based retraining, not a claim that the entire historical hyperparameter search is rerun from scratch. No additional C search is performed.

Expected outputs include `SUMMARY.md`, `cv_summary.csv`, `cv_paired_macro_f1.csv`, `cv_warnings.json`, coefficients, per-image contributions, group metrics and candidate pipelines under the new output folder. Existing output folders are rejected; choose a new run name.

## Recorded Hue #0 comparison


| Candidate                                   | Train accuracy | Validation accuracy | Mean group-CV macro-F1 |
| ------------------------------------------- | -------------- | ------------------- | ---------------------- |
| Full features                               | 92.73%         | 71.43%              | 0.7547                 |
| Remove original feature index 0 and retrain | 96.36%         | 77.14%              | 0.6130                 |

The full model was retained. The higher fixed validation score of the alternative did not outweigh its lower group-CV performance. The Hue #0 hypothesis was developed after inspecting validation errors, so this remains an exploratory comparison.

Zeroing a standardized feature contribution in the saved model is different from removing the input dimension and retraining. The former sets that feature to its training mean; it does not remove coloured pixels from the photograph.

## Analyze without fitting candidates

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python tools/analyze_train_hue0.py </span><span class="token token punctuation">\</span><span>
</span><span>  --mode analyze </span><span class="token token punctuation">\</span><span>
</span><span>  --pipeline models/final/pipeline.joblib </span><span class="token token punctuation">\</span><span>
</span><span>  --dataset-dir dataset_dev </span><span class="token token punctuation">\</span><span>
</span>  --output-dir results/reproduce_analysis_01</code></span></div></figure></pre>

Analysis-only mode does not produce a CV recommendation.

## 8. Reproduce learning curves and explanations

## Whole-location learning curve

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python tools/learning_curve_groups.py </span><span class="token token punctuation">\</span><span>
</span><span>  --pipeline models/final/pipeline.joblib </span><span class="token token punctuation">\</span><span>
</span><span>  --dataset-dir dataset_dev </span><span class="token token punctuation">\</span><span>
</span><span>  --output-dir results/reproduce_learning_curve_01 </span><span class="token token punctuation">\</span><span>
</span><span>  --repeats </span><span class="token token">20</span></code></span></div></figure></pre>

Uses nested, class-stratified subsets of whole training groups with fixed validation. Scaling and classification are refitted for each subset. Exports plots, raw runs, summaries and protocol information. Bands show +/- 1 SD across training-group subsets, not confidence intervals. No test image decoding occurs.

## Active-feature image explanation

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python tools/inspect_active_features.py </span><span class="token token punctuation">\</span><span>
</span><span>  --pipeline models/final/pipeline.joblib </span><span class="token token punctuation">\</span><span>
</span><span>  --dataset-dir dataset_dev </span><span class="token token punctuation">\</span><span>
</span>  --output-dir results/reproduce_active_features_01</code></span></div></figure></pre>

Without `--image-id`, this selects the first validation image. To inspect a specific image, add `--image-id` followed by its exact `image_id` from `images.csv`, not its filename. Train/validation images only are allowed.

Exports `active_features.csv`, `feature_locations.png` and `feature_locations.pdf`. HOG rectangles indicate feature-cell support, not pixel saliency or semantic object recognition. Colour and LBP histograms may be global and cannot be assigned to a unique object.

## 9. Other training and development entry points

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>bash</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>python main.py train --help
</span>python -m src.train --help</code></span></div></figure></pre>

These are generic training/search interfaces, not automatically the exact final logistic-regression configuration. Do not assume their defaults or a center-crop example reproduce the final squash/L1 model. Use Section 7 for the verified final-template retraining interface.

Historical runners include `scripts/run_polyu_lr.py`, `scripts/prepare_polyu_feature_experiment.py` and `scripts/run_feature_experiment.py`. Their configurations can depend on earlier baseline artifacts and output paths that are not part of a minimal submission. Inspect each config and use supported dry-run options before running them; do not treat them as required quick-start steps. For example, `configs/polyu_lr_02.json` references `models_runs/` and `results_runs/`, enables `allow_unverified`, and is a historical development configuration, not the final submission recipe.

## Initial dataset preparation only

The following utilities exist for original data preparation, not routine reproduction of the frozen submission:

* `scripts/update_images_csv.py`: scan/update catalogue entries.
* `scripts/split_dataset.py`: generate grouped splits.
* `tools/organise_photos.py`: create an editable collection manifest and copy new photos into the dataset.

Do not rerun dataset-writing operations or force a new split on the submitted dataset after viewing test results.

## 10. Project layout and optional tools

<pre class="not-prose w-full rounded font-mono text-sm font-light m-0"><figure class="relative flex w-full min-w-0 flex-col rounded-lg border border-soft bg-subtle font-mono font-medium text-primary selection:bg-super-soft selection:text-super-primary [&_.token.deleted]:!text-negative-primary [&_.token.inserted]:!text-positive-primary text-xs"><figcaption class="flex min-h-7 items-center justify-between gap-2 overflow-hidden rounded-t-lg pl-2 pr-1 pt-1 text-xs text-secondary"><span>text</span><div class="ml-auto flex items-center gap-2"><button aria-label="Wrap lines" aria-pressed="false" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft aria-pressed:text-super-primary aria-pressed:bg-super-soft aria-pressed:enabled:hover:bg-super-soft aria-pressed:enabled:hover:text-super-primary"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-text-wrap"></use></svg></button><button aria-label="Copy code" data-state="closed" type="button" class="reset interactable select-none [-webkit-user-drag:none] outline-hidden font-medium transition-[background-color,border-color,color,opacity] duration-300 ease-out font-sans text-center items-center justify-center leading-loose whitespace-nowrap disabled:cursor-default disabled:opacity-50 data-[state=open]:text-primary data-[state=open]:bg-soft h-6 text-xs cursor-pointer inline-flex rounded-full aspect-square p-0 aspect-[9/8] text-secondary hover:text-primary hover:bg-soft"><div class="relative flex items-center justify-center"><div class="inline-flex"><svg role="img" class="inline-flex fill-current shrink-0" width="14" height="14" stroke-width="1.75" aria-hidden="true"><use xlink:href="#pplx-icon-copy"></use></svg></div><div class="absolute inset-0 flex items-center justify-center"></div></div></button></div></figcaption><div class="overflow-hidden rounded-b-lg"><span><code><span>main.py                 unified project entry point
</span>README.md               setup, usage and reproduction instructions
requirements.txt        unpinned dependencies
requirements-lock.txt   recorded environment dependencies
src/                    preprocessing, features, training, app, evaluation, experiments
webui/                  local application HTML/JS/CSS
models/final/           submitted final pipeline and metadata
dataset_dev/            frozen catalogue and supplied/downloaded photographs
configs/                experiment configurations
scripts/                dataset and historical experiment entry points
tools/                  diagnostics and reproduction helpers
tests/                  unit/integration test source
results/                generated/supplied evaluation and analysis records</code></span></div></figure></pre>

The results folder is generated by commands or supplied separately in the submission; it is not necessarily present in Git.


| Tool                                       | Purpose                                                         |
| ------------------------------------------ | --------------------------------------------------------------- |
| `analyze_train_hue0.py`                    | Contributions, Hue #0 ablation, grouped CV and retraining       |
| `learning_curve_groups.py`                 | Whole-location learning curve                                   |
| `inspect_active_features.py`               | Active-feature CSV and image explanation                        |
| `find_duplicates.py`                       | Near-duplicate diagnostics                                      |
| `audit_sources.py`                         | Resolution/compression/sharpness diagnostics                    |
| `nearest_train.py`                         | Feature-space comparison with training images                   |
| `preview_modes.py`                         | Resize-mode inspection                                          |
| `make_variants.py`,`summarise_variants.py` | Controlled image-variant diagnostics                            |
| `organise_photos.py`                       | Optional dataset-development utility; not for normal evaluation |
| `consolidate_tests.py`                     | Optional historical maintenance utility; not a test runner      |

Only the first three tools are required to reproduce the additional analyses described above. Other helpers may be omitted from a cleaned submission if unused; update this table to match the packaged files. Diagnostic variants are not additional independent dataset samples.

## 11. Troubleshooting

* No model found: confirm `models/final/pipeline.joblib` exists and launch with `--outputs-dir models/final`.
* Model/prediction differs after a file replacement: stop and restart the server, check the active model, and compare the exact image/model hashes.
* Evaluation cannot find photos: obtain the full dataset and preserve the catalogue's relative paths.
* Output folder already exists: preserve the previous run and choose a new `--output-dir`.
* Boundary check fails: inspect actual locations and correct the records only after verification; do not bypass checks to obtain a final score.
* Saved-model loading fails: check Python/package versions and that `src/` is importable from the project root.
* Convergence warnings: inspect diagnostic files; do not silently treat affected candidates as validated.

## 12. Submission and AI-use transparency

Submit the report PDF separately from the correctly named code ZIP. Include the dataset or a working `DATASET_LINK.txt`. Keep the final model, relevant source, configs, tests, experiment evidence and instructions. Exclude virtual environments, caches, credentials, Git internals and redundant temporary artifacts.

Generative AI assisted implementation/debugging and analysis tooling. The report should identify the assistance, observed mistakes and the student's verification/corrections, including the application-state discrepancy and the distinction between a fixed-model ablation and retraining. Do not claim a fresh-machine run, complete geographic audit or other verification until it has actually been performed.
