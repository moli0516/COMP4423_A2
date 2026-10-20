import argparse
import sys
from pathlib import Path

from .camera_config import CameraConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live camera preview with periodic saved-pipeline predictions.")
    parser.add_argument("--model", type=Path, required=True, help="Trusted full pipeline.joblib from current source.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0, help="Target minimum seconds between accepted samples.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--backend", choices=("auto", "any", "avfoundation"), default="auto")
    parser.add_argument("--mirror-preview", action="store_true", help="Mirror only the display, never the inference image.")
    args = parser.parse_args(argv)
    try:
        config = CameraConfig(
            model_path=args.model, camera_index=args.camera_index,
            interval_seconds=args.interval, width=args.width, height=args.height,
            backend=args.backend, mirror_preview=args.mirror_preview,
        )
        from .camera_application import CameraApplication
        return CameraApplication(config).run()
    except KeyboardInterrupt:
        print("\nCamera stopped.")
        return 130
    except ImportError as error:
        print(f"ERROR: {error}. Install the project's ML dependencies and opencv-python (GUI build).", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
