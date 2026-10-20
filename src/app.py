"""Local web interface for the PolyU inside/outside campus classifier (Task 5).

Launch from the project root (zero extra dependencies beyond the training stack):

    python -m src.app                      # then open http://127.0.0.1:8000
    python -m src.app --port 8080 --no-browser

Design notes
- The server only passes the raw image bytes to the saved scikit-learn pipeline, so the
  exact same preprocessing and feature extraction as in evaluation is used. File names,
  folder names, EXIF location data and GPS are never read or used.
- Models are discovered by scanning ``--outputs-dir`` for ``pipeline.joblib`` files;
  the browser can only choose from that discovered list (no arbitrary paths).
- The dataset page reads ``images.csv`` / ``locations.csv`` only. It never opens image
  files, so the held-out test split stays untouched.
- The server binds to 127.0.0.1 by default. ``joblib.load`` can execute code, so only
  serve pipelines you trained yourself.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # lets joblib import src.features when run as a script

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset_dev"
WEB_DIR = PROJECT_ROOT / "webui"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
LABEL_TEXT = {1: "Inside PolyU main campus", 0: "Outside PolyU main campus"}
SPLIT_ORDER = ("train", "validation", "test")
PAGES = {"/": "index.html", "/config": "config.html"}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_plain(value):
    """Convert config objects and numpy scalars to JSON-friendly values."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def is_best(model_id: str) -> bool:
    return "best" in model_id.split("/")


def describe_pipeline(pipeline) -> dict:
    """List pipeline steps with their (non-deep) parameters, for the config page."""
    steps = []
    for name, step in getattr(pipeline, "steps", []):
        params = {}
        if hasattr(step, "get_params"):
            params = {key: to_plain(val) for key, val in step.get_params(deep=False).items()}
        steps.append({"name": name, "type": type(step).__name__, "params": params})
    return {"steps": steps}


def score_inside(classifier, features) -> tuple[float | None, str | None]:
    """Auxiliary confidence value; the label itself always comes from classifier.predict."""
    if hasattr(classifier, "decision_function"):
        value = np.asarray(classifier.decision_function(features)).ravel()
        if value.size == 1:
            return float(value[0]), "decision score (above 0 means inside)"
    if hasattr(classifier, "predict_proba") and hasattr(classifier, "classes_"):
        classes = list(classifier.classes_)
        if 1 in classes:
            proba = classifier.predict_proba(features)[0]
            return float(proba[classes.index(1)]), "probability of inside (above 0.5 means inside)"
    return None, None


class ModelStore:
    """Discovers saved pipelines and keeps the active one loaded."""

    def __init__(self, outputs_dir: Path):
        self.outputs_dir = Path(outputs_dir).resolve()
        self._lock = threading.RLock()
        self._paths: dict[str, Path] = {}
        self._cache: dict[str, object] = {}
        self.active_id: str | None = None
        self.refresh()

    @property
    def state_file(self) -> Path:
        return self.outputs_dir / ".app_state.json"

    def refresh(self) -> None:
        with self._lock:
            found: dict[str, Path] = {}
            if self.outputs_dir.is_dir():
                for path in self.outputs_dir.rglob("pipeline.joblib"):
                    found[path.parent.relative_to(self.outputs_dir).as_posix()] = path
            ordered = sorted(found.items(), key=lambda item: (not is_best(item[0]), item[0]))
            self._paths = dict(ordered)
            self._cache = {key: val for key, val in self._cache.items() if key in self._paths}
            if self.active_id not in self._paths:
                saved = read_json(self.state_file).get("active_model")
                self.active_id = saved if saved in self._paths else next(iter(self._paths), None)

    def metadata(self, model_id: str) -> dict:
        return read_json(self._paths[model_id].with_name("pipeline_metadata.json"))

    def list_models(self) -> list[dict]:
        with self._lock:
            items = []
            for model_id, path in self._paths.items():
                meta = self.metadata(model_id)
                variant = meta.get("variant") if isinstance(meta.get("variant"), dict) else {}
                items.append({
                    "id": model_id,
                    "is_best": is_best(model_id),
                    "variant_id": variant.get("variant_id"),
                    "trained_on_splits": meta.get("trained_on_splits"),
                    "selected_using_splits": meta.get("selected_using_splits"),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                })
            return items

    def load(self, model_id: str):
        with self._lock:
            if model_id not in self._paths:
                raise KeyError(model_id)
            if model_id not in self._cache:
                self._cache[model_id] = joblib.load(self._paths[model_id])
            return self._cache[model_id]

    def select(self, model_id: str) -> None:
        self.load(model_id)  # fail before switching if the file cannot be loaded
        with self._lock:
            self.active_id = model_id
            try:
                self.state_file.write_text(json.dumps({"active_model": model_id}), encoding="utf-8")
            except OSError:
                pass  # remembering the choice is a convenience only

    def active_details(self) -> dict:
        with self._lock:
            if self.active_id is None:
                return {"active": None}
            pipeline = self.load(self.active_id)
            return {
                "active": self.active_id,
                "metadata": self.metadata(self.active_id),
                "pipeline": describe_pipeline(pipeline),
            }

    def predict(self, data: bytes) -> dict:
        with self._lock:
            model_id = self.active_id
            if model_id is None:
                raise LookupError("No trained pipeline found. Run training first.")
            pipeline = self.load(model_id)
        started = time.perf_counter()
        # Same steps as Pipeline.predict, but features are extracted only once.
        features = pipeline[:-1].transform([data])
        classifier = pipeline[-1]
        label = int(classifier.predict(features)[0])
        score, score_kind = score_inside(classifier, features)
        return {
            "label": label,
            "text": LABEL_TEXT[label],
            "score": score,
            "score_kind": score_kind,
            "model": model_id,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }


def summarise_dataset(dataset_dir: Path, model_meta: dict | None = None) -> dict:
    images_csv = Path(dataset_dir) / "images.csv"
    locations_csv = Path(dataset_dir) / "locations.csv"
    if not images_csv.is_file():
        return {"available": False, "message": f"{images_csv} was not found."}
    images = read_csv(images_csv)
    locations = {row["group_id"]: row for row in read_csv(locations_csv)} if locations_csv.is_file() else {}

    image_counts: dict[tuple, int] = defaultdict(int)
    group_images: dict[str, int] = defaultdict(int)
    group_info: dict[str, tuple] = {}
    for row in images:
        name = "inside" if str(row.get("label")).strip() == "1" else "outside"
        split = row.get("split", "")
        image_counts[(split, name)] += 1
        group_images[row["group_id"]] += 1
        group_info[row["group_id"]] = (split, name)

    group_counts: dict[tuple, int] = defaultdict(int)
    for split, name in group_info.values():
        group_counts[(split, name)] += 1

    splits = sorted({key[0] for key in image_counts}, key=lambda s: (SPLIT_ORDER.index(s) if s in SPLIT_ORDER else 99, s))
    summary = []
    for split in splits:
        entry = {"split": split}
        for name in ("inside", "outside"):
            entry[name] = {"images": image_counts[(split, name)], "groups": group_counts[(split, name)]}
        entry["total_images"] = entry["inside"]["images"] + entry["outside"]["images"]
        summary.append(entry)

    groups = []
    for group_id, (split, name) in sorted(group_info.items(), key=lambda item: (item[1][1], item[1][0], item[0])):
        loc = locations.get(group_id, {})
        groups.append({
            "group_id": group_id, "class": name, "split": split, "images": group_images[group_id],
            "description": loc.get("location_description", ""),
            "boundary_verified": loc.get("boundary_verified", ""),
        })

    digest = hashlib.sha256(images_csv.read_bytes()).hexdigest()
    trained_digest = (model_meta or {}).get("images_csv_sha256")
    return {
        "available": True,
        "dataset_dir": str(dataset_dir),
        "summary": summary,
        "total_images": len(images),
        "total_groups": len(group_info),
        "images_csv_sha256": digest,
        "matches_active_model": (digest == trained_digest) if trained_digest else None,
        "groups": groups,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "PolyUCampusApp/1.0"

    @property
    def store(self) -> ModelStore:
        return self.server.store

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind in ("application/javascript", "application/json"):
            kind += "; charset=utf-8"
        self._send(HTTPStatus.OK, path.read_bytes(), kind)

    def _read_body(self) -> bytes | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._error(HTTPStatus.BAD_REQUEST, "Empty request body.")
            return None
        if length > MAX_UPLOAD_BYTES:
            self.close_connection = True
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Image is larger than 25 MB.")
            return None
        return self.rfile.read(length)

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        path = self.path.split("?", 1)[0]
        web_dir = self.server.web_dir.resolve()
        if path in PAGES:
            self._file(web_dir / PAGES[path])
        elif path.startswith("/static/"):
            candidate = (web_dir / path[len("/static/"):]).resolve()
            if candidate.parent != web_dir:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
            else:
                self._file(candidate)
        elif path == "/api/models":
            self.store.refresh()
            self._json({"active": self.store.active_id, "models": self.store.list_models()})
        elif path == "/api/models/active":
            try:
                self._json(self.store.active_details())
            except Exception as error:  # unreadable pickle, missing import, ...
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Could not load the model: {error}")
        elif path == "/api/dataset":
            meta = self.store.metadata(self.store.active_id) if self.store.active_id else {}
            self._json(summarise_dataset(self.server.dataset_dir, meta))
        else:
            self._error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        body = self._read_body()
        if body is None:
            return
        if path == "/api/predict":
            try:
                self._json(self.store.predict(body))
            except LookupError as error:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            except ValueError as error:  # includes PreprocessingError for undecodable images
                self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"Could not read that image: {error}")
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Prediction failed: {error}")
        elif path == "/api/models/select":
            try:
                model_id = json.loads(body.decode("utf-8")).get("id")
                self.store.select(model_id)
                self._json({"active": self.store.active_id})
            except (KeyError, ValueError, AttributeError):
                self._error(HTTPStatus.BAD_REQUEST, "Unknown model id.")
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Could not load the model: {error}")
        else:
            self._error(HTTPStatus.NOT_FOUND, "Not found")


def create_server(host: str, port: int, outputs_dir: Path, dataset_dir: Path, web_dir: Path = WEB_DIR) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.store = ModelStore(outputs_dir)
    server.dataset_dir = Path(dataset_dir)
    server.web_dir = Path(web_dir)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local web UI for the inside/outside campus classifier.")
    parser.add_argument("--host", default="127.0.0.1", help="Keep 127.0.0.1 so the app stays local.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR, help="Folder scanned for pipeline.joblib files.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    try:
        server = create_server(args.host, args.port, args.outputs_dir, args.dataset_dir)
    except OSError as error:
        print(f"ERROR cannot start server on {args.host}:{args.port}: {error}", file=sys.stderr)
        return 1
    url = f"http://{args.host}:{server.server_address[1]}"
    if server.store.active_id is None:
        print(f"WARNING no pipeline.joblib found under {server.store.outputs_dir}; train a model first.")
    else:
        print(f"Active model: {server.store.active_id}")
    print(f"Serving on {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
