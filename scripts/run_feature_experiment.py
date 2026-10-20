"""Run the frozen POLYU-FEATURE-01 plan; prepare/freeze it first."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.feature_ablation.feature_experiment_config import FeatureExperimentConfig
from src.experiments.feature_ablation.feature_experiment_runner import FeatureExperimentRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/polyu_feature_01.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/polyu_feature_01/run_01")
    parser.add_argument("--dry-run", action="store_true", help="Verify the frozen plan and inputs without fitting.")
    parser.add_argument(
        "--confirm-camera-diagnostic-completed", action="store_true",
        help="Record that you completed the saved-image versus camera-predictor diagnostic.")
    args = parser.parse_args(argv)
    try:
        config = FeatureExperimentConfig.from_json(args.config, ROOT)
        output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        runner = FeatureExperimentRunner(
            config, output, camera_diagnostic_confirmed=args.confirm_camera_diagnostic_completed)
        result = runner.run(dry_run=args.dry_run)
        if not args.dry_run:
            print(f"Comparison: {Path(output).resolve() / 'comparison.csv'}")
            print(f"Best model: {Path(output).resolve() / 'best/pipeline.joblib'}")
        return 0 if result else 1
    except KeyboardInterrupt:
        print("Interrupted. Preserve outputs and inspect run_manifest.json.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
