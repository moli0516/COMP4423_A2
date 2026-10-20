import csv
import io
import json
from pathlib import Path

import joblib

from src.utils.file_io import write_atomic
from src.utils.serialization import to_python


class ArtifactStore:
    """Serialise all artifacts first; replace each destination individually."""

    def __init__(self, models_dir, results_dir):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)

    def save(self, pipeline, metadata, report, trials):
        model = io.BytesIO()
        joblib.dump(pipeline, model)
        metadata_text = json.dumps(to_python(metadata), indent=2, ensure_ascii=False, allow_nan=False)
        report_text = json.dumps(to_python(report), indent=2, ensure_ascii=False, allow_nan=False)
        table = io.StringIO(newline="")
        writer = csv.DictWriter(table, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
        outputs = [
            (self.models_dir / "pipeline.joblib", model.getvalue(), True),
            (self.models_dir / "pipeline_metadata.json", metadata_text, False),
            (self.results_dir / "validation_experiments.csv", table.getvalue(), False),
            (self.results_dir / "training_report.json", report_text, False),
        ]
        for path, content, binary in outputs:
            write_atomic(path, lambda handle, value=content: handle.write(value), binary=binary)
