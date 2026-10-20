"""Prepare/freeze F0-F3 feature-ablation architecture; never train a model."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.feature_ablation.feature_experiment_config import FeatureExperimentConfig
from src.experiments.feature_ablation.feature_experiment_planner import FeatureExperimentPlanner
from src.experiments.feature_ablation.experiment_plan_store import ExperimentPlanStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/polyu_feature_01.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Validate and print the plan without writing (default).")
    action.add_argument("--freeze", action="store_true", help="Save a new plan and CSV snapshots without fitting.")
    args = parser.parse_args(argv)
    try:
        config = FeatureExperimentConfig.from_json(args.config, ROOT)
        prepared = FeatureExperimentPlanner(config).prepare()
        for variant in prepared["plan"]["variants"]:
            print(f"{variant['variant_id']}: {variant['description']}; dimension={variant['feature_dimension']}")
        print("Estimator: fixed L2 LR C=0.01; split: unchanged; completed fits: 0")
        print(f"Camera diagnostic status: {config.camera_diagnostic_status}")
        if args.freeze:
            output = ExperimentPlanStore(config.plan_dir).freeze(prepared)
            print(f"Prepared plan saved to {output}. No training was performed.")
        else:
            print("DRY RUN: no files written, no image decoding, no fitting.")
        return 0
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
