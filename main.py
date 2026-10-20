"""Project entry point; core logic lives in Application."""
from pathlib import Path

from src.application import Application

ROOT = Path(__file__).resolve().parent


def run_module(arguments):
    return Application(root=ROOT).run_module(arguments)


def run_tests(arguments=None):
    return Application(root=ROOT).run_tests(arguments)


def run_train(arguments):
    return Application(root=ROOT).run_train(arguments)


def make_parser():
    return Application.make_parser()


def main(argv=None):
    return Application(root=ROOT).run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
