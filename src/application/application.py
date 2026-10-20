import argparse
import subprocess
import sys
from pathlib import Path


class Application:
    """Unified CLI orchestration with camera support and lazy ML imports."""

    def __init__(self, root=None, python_executable=None):
        self.root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
        self.python_executable = python_executable or sys.executable

    def run_module(self, arguments):
        command = [self.python_executable, "-m", *arguments]
        print("Running:", " ".join(command), flush=True)
        try:
            return subprocess.run(command, cwd=self.root, check=False).returncode
        except OSError as error:
            print(f"ERROR: Could not start Python: {error}", file=sys.stderr)
            return 1

    def run_tests(self, arguments=None):
        parser = argparse.ArgumentParser(prog="python main.py test")
        parser.add_argument("--pattern", default="test*.py")
        parser.add_argument("--failfast", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        options = parser.parse_args(arguments)
        if not (self.root / "tests").is_dir():
            print("ERROR: Missing tests directory.", file=sys.stderr)
            return 1
        arguments = [
            "unittest", "discover", "-s", "tests", "-t", ".", "-p", options.pattern,
            "-q" if options.quiet else "-v",
        ]
        if options.failfast:
            arguments.append("-f")
        return self.run_module(arguments)

    def run_train(self, arguments):
        if not (self.root / "src" / "train.py").is_file():
            print("ERROR: Missing src/train.py.", file=sys.stderr)
            return 1
        return self.run_module(["src.train", *arguments])

    def run_camera(self, arguments):
        if not (self.root / "src" / "camera" / "__main__.py").is_file():
            print("ERROR: Missing src/camera/__main__.py.", file=sys.stderr)
            return 1
        return self.run_module(["src.camera", *arguments])

    @staticmethod
    def make_parser():
        return argparse.ArgumentParser(
            prog="python main.py", description="Unified project entry point.",
            epilog=(
                "Commands:\n"
                "  test    Run unittest discovery.\n"
                "  train   Forward remaining arguments to src.train.\n"
                "  all     Run tests; train only if they pass.\n"
                "  camera  Live preview with periodic saved-pipeline predictions.\n\n"
                "Examples:\n"
                "  python main.py test\n"
                "  python main.py train --help\n"
                "  python main.py camera --model models_oop_verified/pipeline.joblib\n\n"
                "Relative data/model paths resolve from the project root. "
                "Existing training outputs may be overwritten."
            ), formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    def run(self, argv=None):
        argv = list(sys.argv[1:] if argv is None else argv)
        parser = self.make_parser()
        if not argv or argv[0] in {"-h", "--help"}:
            parser.print_help()
            return 0
        command, arguments = argv[0], argv[1:]
        if command not in {"test", "train", "all", "camera"}:
            parser.error(f"unknown command: {command!r}; choose test, train, all, or camera")
        try:
            if command == "test":
                return self.run_tests(arguments)
            if command == "camera":
                return self.run_camera(arguments)
            if command == "train" or "--help" in arguments or "-h" in arguments:
                return self.run_train(arguments)
            code = self.run_tests([])
            if code != 0:
                print("Tests did not pass. Training was not started.", file=sys.stderr)
                return code
            print("Tests passed. Starting training.", flush=True)
            return self.run_train(arguments)
        except KeyboardInterrupt:
            print("\nCancelled by user.", file=sys.stderr)
            return 130
