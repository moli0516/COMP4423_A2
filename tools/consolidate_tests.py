#!/usr/bin/env python3
"""Move the root-level *_checks test packages into tests/ so one command runs every test.

    python tools/consolidate_tests.py            # preview only, nothing is changed
    python tools/consolidate_tests.py --apply    # git mv + rewrite imports and docs

    camera_checks              -> tests/camera
    feature_ablation_checks    -> tests/feature_ablation
    feature_experiment_checks  -> tests/feature_experiment
    oop_checks                 -> tests/oop
    polyu_lr_checks            -> tests/polyu_lr

If you already moved the folders by hand to tests/<name>_checks, keep those names and only fix the imports:

    python tools/consolidate_tests.py --keep-names --allow-dirty            # preview
    python tools/consolidate_tests.py --keep-names --allow-dirty --apply

Inside the moved files and anywhere else in the repo, `camera_checks.helpers` becomes
`tests.camera.helpers` (or `tests.camera_checks.helpers` with --keep-names), and `-s camera_checks`
becomes `-s tests/camera`. Folders moved from the project root also get `Path(__file__)...parents[N]`
bumped to N+1 because they are one level deeper. Re-running is safe: nothing is rewritten twice.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

RENAMES = {
    "camera_checks": "camera",
    "feature_ablation_checks": "feature_ablation",
    "feature_experiment_checks": "feature_experiment",
    "oop_checks": "oop",
    "polyu_lr_checks": "polyu_lr",
}
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "_archive", "dataset_dev", "outputs", "temp", "node_modules"}
CODE_SUFFIXES = {".py"}
TEXT_SUFFIXES = {".md", ".txt", ".sh", ".toml", ".cfg", ".ini", ".yml", ".yaml"}
MAX_BYTES = 1_000_000

OLD = "|".join(re.escape(name) for name in RENAMES)
CLI = re.compile(rf"(-s\s+)(?:\./)?({OLD})(?=[\s/'\"]|$)")
MODULE = re.compile(rf"(?<![\w./-])({OLD})\b")
PARENTS = re.compile(r"(__file__[^\n]*?\.parents\[)(\d+)(\])")


def bump_parents(text: str) -> tuple[str, int]:
    count = 0

    def bump(match):
        nonlocal count
        count += 1
        return f"{match.group(1)}{int(match.group(2)) + 1}{match.group(3)}"

    return PARENTS.sub(bump, text), count


def rewrite(text: str, suffix: str, names: dict[str, str]) -> tuple[str, int]:
    count = 0

    def cli(match):
        nonlocal count
        count += 1
        return f"{match.group(1)}tests/{names[match.group(2)]}"

    def module(match):
        nonlocal count
        count += 1
        joiner = "." if suffix in CODE_SUFFIXES else "/"
        return f"tests{joiner}{names[match.group(1)]}"

    text = CLI.sub(cli, text)
    return MODULE.sub(module, text), count


def repo_files(root: Path):
    this_file = Path(__file__).resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in CODE_SUFFIXES | TEXT_SUFFIXES or path.resolve() == this_file:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts) or path.stat().st_size > MAX_BYTES:
            continue
        yield path


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="actually move and rewrite (default: preview)")
    parser.add_argument("--allow-dirty", action="store_true", help="skip the clean-working-tree check")
    parser.add_argument("--keep-names", action="store_true", help="keep the *_checks names (tests/camera_checks, ...)")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()

    if not (root / "tests").is_dir():
        print("ERROR: run from the project root (no tests/ folder here).", file=sys.stderr)
        return 1
    in_git = git(root, "rev-parse", "--is-inside-work-tree").stdout.strip() == "true"
    if args.apply and in_git and not args.allow_dirty and git(root, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        print("ERROR: the git working tree is not clean. Commit or stash first (or use --allow-dirty).", file=sys.stderr)
        return 1

    names = {old: (old if args.keep_names else new) for old, new in RENAMES.items()}
    moves = []  # (source dir, destination dir, moved from the project root?)
    for old, final in names.items():
        destination = root / "tests" / final
        if (root / old).is_dir():
            moves.append((root / old, destination, True))
        elif (root / "tests" / old).is_dir() and destination != root / "tests" / old:
            moves.append((root / "tests" / old, destination, False))
    clashes = [str(dst.relative_to(root)) for _, dst, _ in moves if dst.exists()]
    if clashes:
        print(f"ERROR: destination already exists: {', '.join(clashes)}", file=sys.stderr)
        return 1
    missing_init = [src.name for src, _, _ in moves if not (src / "__init__.py").is_file()]
    if missing_init:
        print(f"WARNING: no __init__.py in {', '.join(missing_init)}; unittest discovery needs one.")

    print(f"{'APPLY' if args.apply else 'PREVIEW'} (git repository: {in_git}, keep names: {args.keep_names})")
    if not moves:
        print("No folders to move; only references will be checked.")
    for src, dst, _ in moves:
        print(f"  MOVE  {src.relative_to(root)}/  ->  {dst.relative_to(root)}/")

    edits = []
    for path in repo_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new_text, count = rewrite(text, path.suffix, names)
        bumped = 0
        if path.suffix in CODE_SUFFIXES and any(src in path.parents for src, _, from_root in moves if from_root):
            new_text, bumped = bump_parents(new_text)
        if count or bumped:
            edits.append((path, new_text, count, bumped))
    for path, _, count, bumped in edits:
        extra = f", {bumped} __file__ depth fix{'es' if bumped != 1 else ''}" if bumped else ""
        print(f"  EDIT  {path.relative_to(root)}  ({count} reference{'s' if count != 1 else ''}{extra})")
    if not moves and not edits:
        print("Nothing to do.")
        return 0

    if not args.apply:
        print("\nNothing was changed. Re-run with --apply to perform it.")
        return 0

    for src, dst, _ in moves:
        if in_git and git(root, "mv", str(src.relative_to(root)), str(dst.relative_to(root))).returncode == 0:
            continue
        shutil.move(str(src), str(dst))
    for path, new_text, _, _ in edits:
        target = path
        for src, dst, _ in moves:  # the edited file may have just been moved
            if src in path.parents:
                target = dst / path.relative_to(src)
        target.write_text(new_text, encoding="utf-8")
    print("\nDone. Next:\n  python -m unittest discover -s tests -t . -q\n  git status --short | head -30")
    print("Also check by hand any test that finds the project root another way than .parents[N].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())