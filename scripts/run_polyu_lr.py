"""Run POLYU-LR-01 inside the PolyU assignment_2 project, not the tabular repo."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.polyu_lr_config import PolyuLRConfig
from src.experiments.polyu_holdout_experiment import PolyuHoldoutExperiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/polyu_lr_01.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = PolyuLRConfig.from_json(args.config, ROOT)
        PolyuHoldoutExperiment(config).run(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("Interrupted. Preserve the output directory and inspect its manifest.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
