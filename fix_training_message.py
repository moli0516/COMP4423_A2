"""Restore the three error-message contracts without modifying tests.

Run from the project root: python fix_training_messages.py
Both files are checked before any write. Originals are backed up once.
"""
from pathlib import Path
import ast


def main():
    root = Path.cwd()
    changes = {
        "src/data/dataset_catalog.py": [
            ("duplicate image path", "duplicate relative_path"),
            (
                'f"{place}: empty {name}; assign all splits before training."',
                '(f"{place}: empty split. Run scripts/split_dataset.py first." '
                'if name == "split" else f"{place}: empty {name}.")',
            ),
        ],
        "src/utils/file_io.py": [
            ("Image path leaves dataset directory:", "Image path leaves the dataset directory:"),
        ],
    }
    pending = []
    for relative, replacements in changes.items():
        path = root / relative
        source = path.read_text(encoding="utf-8")
        updated = source
        for old, new in replacements:
            count = updated.count(old)
            if count == 1:
                updated = updated.replace(old, new, 1)
            elif count == 0 and new in updated:
                continue
            else:
                raise RuntimeError(
                    f"{relative}: expected exactly one target or an already-applied fix; "
                    f"found {count} matches for {old!r}. No files have been changed."
                )
        ast.parse(updated, filename=relative)
        compile(updated, relative, "exec")
        pending.append((path, source, updated))
    for path, source, updated in pending:
        if source == updated:
            print(f"Already fixed: {path.relative_to(root)}")
            continue
        backup = path.with_name(path.name + ".before_message_fix.bak")
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}; inspect it before retrying.")
    for path, source, updated in pending:
        if source == updated:
            continue
        backup = path.with_name(path.name + ".before_message_fix.bak")
        backup.write_text(source, encoding="utf-8")
        path.write_text(updated, encoding="utf-8")
        print(f"Fixed: {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())