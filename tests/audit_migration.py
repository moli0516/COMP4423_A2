"""Static migration audit with Python 3.12/3.13 dump compatibility."""
import ast
import csv
import hashlib
import inspect
from pathlib import Path


def baseline_ast_dump(node):
    """Preserve the Python 3.12 representation used by the original manifest."""
    options = {"include_attributes": False}
    if "show_empty" in inspect.signature(ast.dump).parameters:
        options["show_empty"] = True
    return ast.dump(node, **options)


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "TEST_MIGRATION_MAP.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 142:
        raise AssertionError(f"Expected 142 manifest entries, found {len(rows)}")
    expected = {
        (row["new_file"], row["new_class"], row["new_method"]): row
        for row in rows
    }
    if len(expected) != 142:
        raise AssertionError("Duplicate test entries in migration manifest.")
    found = set()
    assertions = 0
    mismatches = []
    for path in sorted((root / "tests").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        classes = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ClassDef)]
        if len(classes) > 1:
            raise AssertionError(f"More than one class: {path}")
        relative = path.relative_to(root).as_posix()
        for cls in classes:
            for method in cls.body:
                if not isinstance(method, ast.FunctionDef) or not method.name.startswith("test_"):
                    continue
                key = (relative, cls.name, method.name)
                if key not in expected:
                    raise AssertionError(f"Unexpected or duplicate legacy test: {key}")
                if key in found:
                    raise AssertionError(f"Duplicate test definition: {key}")
                row = expected[key]
                digest = hashlib.sha256(baseline_ast_dump(method).encode("utf-8")).hexdigest()
                if digest != row["method_ast_sha256"]:
                    mismatches.append(
                        f"{key}\n  expected: {row['method_ast_sha256']}\n  actual:   {digest}"
                    )
                count = sum(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr.startswith("assert")
                    for node in ast.walk(method)
                )
                if count != int(row["assertion_calls"]):
                    raise AssertionError(f"Assertion count differs: {key}")
                assertions += count
                found.add(key)
    if found != set(expected):
        raise AssertionError(f"Missing tests: {sorted(set(expected) - found)}")
    if mismatches:
        raise AssertionError(
            "Test bodies still differ after dump-format compatibility handling. "
            "Do not regenerate the baseline automatically.\n"
            + "\n".join(mismatches)
        )
    print(f"Static migration audit passed: {len(found)} test methods; {assertions} assertion calls.")
    print("This is not a runtime pass result. Run unittest discovery separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())