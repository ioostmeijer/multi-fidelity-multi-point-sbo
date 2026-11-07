"""Based on claude generated file.

Auto-generates the Constants Reference page by scanning constants/.

Handles four constant shapes automatically:
  - dict[str, float/int/str]  → two-column table (Name | Value)
  - dict[str, list]           → dynamic column table (one col per list element)
  - list                      → bullet list
  - combination ({**A, **B})  → "combines X and Y" notice + inherited table
  - empty list/dict           → explicit "currently empty" notice

dict[str, list[Any]] tables have custom headers. These are defined in the variable
LIST_COLUMN_HEADERS. If such an entry is added update this variable with the new
column names.
"""

import ast
import importlib
import inspect
import sys
from collections import defaultdict
from pathlib import Path

import mkdocs_gen_files

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src"
PACKAGE = SRC / "sbo"
CONSTANTS_ROOT = PACKAGE / "constants"

sys.path.insert(0, str(SRC))

# Map variable name patterns to human-readable column headers for list values.
# Key: substring that appears in the variable name (case-insensitive)
# Value: list of column headers matching the list positions
LIST_COLUMN_HEADERS: dict[str, list[str]] = {
    "FUNCTION_DELAY": ["HF delay (s)", "LF delay (s)"],
}


def _list_headers(var_name: str, list_len: int) -> list[str]:
    """Return column headers for a list value, or generic [0], [1], ... fallback."""
    var_upper = var_name.upper()
    for pattern, headers in LIST_COLUMN_HEADERS.items():
        if pattern in var_upper and len(headers) == list_len:
            return headers
    return [f"[{i}]" for i in range(list_len)]


def _parse_combinations(py_file: Path) -> dict[str, list[str] | None]:
    """Return {var_name: [source_var, ...] | None}.

    None  → defined from scratch (literal dict or list)
    [..] → combines or aliases these named variables
    """
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            var = target.id
            val = node.value

            # X = Y  (direct alias)
            if isinstance(val, ast.Name):
                result[var] = [val.id]

            # X = {**A, **B, "extra": ...}
            elif isinstance(val, ast.Dict):
                parents = []
                for k, v in zip(val.keys, val.values):
                    if k is None and isinstance(v, ast.Name):  # **spread
                        parents.append(v.id)
                result[var] = parents if parents else None

            # X = [*A, *B] or plain list
            elif isinstance(val, ast.List):
                parents = []
                for elt in val.elts:
                    if isinstance(elt, ast.Starred) and isinstance(elt.value, ast.Name):
                        parents.append(elt.value.id)
                result[var] = parents if parents else None

            else:
                result[var] = None

    return result


def _is_public_constant(name: str, value: dict | list | set | tuple) -> bool:
    """True for UPPER_CASE names holding dict, list, set, tuple (not dunder)."""
    if name.startswith("_"):
        return False
    if not name.isupper() and not any(c == "_" for c in name):
        return False
    # Accept only the data types we know how to render
    return isinstance(value, (dict, list, set, tuple))


def _collect_constants() -> tuple[list[dict], dict[str, list[str] | None]]:
    """Return (records, combinations).

    records: list of {var_name, display, category, value, module}
    combinations: {var_name: [source_vars] | None}
    """
    records = []
    combinations = {}

    for py_file in sorted(CONSTANTS_ROOT.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue

        mod_name = ".".join(py_file.relative_to(SRC).with_suffix("").parts)
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f"[gen_constants] Skipping {mod_name}: {e}")
            continue

        # Subfolder after constants/ = category
        parts = py_file.relative_to(CONSTANTS_ROOT).parts
        category = parts[0].replace("_", " ").title() if len(parts) > 1 else "General"
        file_stem = py_file.stem  # e.g. "known_optima"

        combinations.update(_parse_combinations(py_file))

        for var_name, value in inspect.getmembers(mod):
            # Only variables defined in *this* module
            if not _is_public_constant(var_name, value):
                continue
            if getattr(sys.modules[mod_name], var_name, None) is None:
                continue

            display = var_name.replace("_", " ").title()
            records.append(
                {
                    "var_name": var_name,
                    "display": display,
                    "category": category,
                    "file": file_stem,
                    "value": value,
                    "module": mod_name,
                }
            )

    return records, combinations


def _render_scalar_dict(d: dict) -> list[str]:
    """dict[str, scalar] → two-column Name | Value table."""
    lines = [
        "| Name | Value |\n",
        "|------|-------|\n",
    ]
    for k, v in sorted(d.items()):
        lines.append(f"| `{k}` | `{v!r}` |\n")
    return lines


def _render_list_dict(var_name: str, d: dict) -> list[str]:
    """dict[str, list] → dynamic column table."""
    # Infer number of columns from first value
    first_val = next(iter(d.values()))
    n_cols = len(first_val)
    col_headers = _list_headers(var_name, n_cols)

    header = "| Name | " + " | ".join(col_headers) + " |\n"
    sep = "|------|" + "|".join(["------"] * n_cols) + "|\n"
    lines = [header, sep]

    for k, v in sorted(d.items()):
        cols = " | ".join(f"`{x!r}`" for x in v)
        lines.append(f"| `{k}` | {cols} |\n")
    return lines


def _render_list(value: list) -> list[str]:
    """Plain list → bullet list or empty notice."""
    if not value:
        return ["*Currently empty — no entries defined.*\n"]
    return [f"- `{item!r}`\n" for item in value]


def _render_value(
    var_name: str, value: dict | list | set | tuple | str | int | float
) -> list[str]:
    """Dispatch to the right renderer based on value shape."""
    if isinstance(value, dict):
        if not value:
            return ["*Currently empty — no entries defined.*\n"]
        first_val = next(iter(value.values()))
        if isinstance(first_val, (list, tuple)):
            return _render_list_dict(var_name, value)
        else:
            return _render_scalar_dict(value)
    elif isinstance(value, (list, set, tuple)):
        return _render_list(list(value))
    return [f"`{value!r}`\n"]


def _combination_notice(
    sources: list[str], var_to_display: dict[str, str]
) -> list[str]:
    if len(sources) == 1:
        src = sources[0]
        disp = var_to_display.get(src, src)
        return [
            f'!!! info "Alias"\n'
            f"    This constant is identical to **{disp}**. "
            f"Shown here for convenience.\n\n"
        ]
    source_links = ", ".join(f"**{var_to_display.get(s, s)}**" for s in sources)
    return [
        f'!!! info "Combined constant"\n'
        f"    This constant merges {source_links} into a single lookup. "
        f"Refer to the source constants for full detail.\n\n"
    ]


def main() -> None:
    """Gather all constants and render the reference page."""
    records, combinations = _collect_constants()

    var_to_display = {r["var_name"]: r["display"] for r in records}

    # Group by category, then by file within category
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        groups[rec["category"]][rec["file"]].append(rec)

    lines = [
        "# Constants Reference\n\n",
        "Auto-generated from all public constants in `constants/`. "
        "Add a new file and it appears here on the next build.\n\n",
    ]

    for category, files in sorted(groups.items()):
        lines.append(f"## {category}\n\n")

        for file_stem, recs in sorted(files.items()):
            # File-level subheading only if multiple files in category
            if len(files) > 1:
                lines.append(f"### {file_stem.replace('_', ' ').title()}\n\n")

            for rec in recs:
                var = rec["var_name"]
                lines.append(f"#### `{var}`\n\n")

                sources = combinations.get(var)

                if sources:
                    # Show combination/alias notice
                    lines.extend(_combination_notice(sources, var_to_display))
                    # Still render the merged value for quick reference
                    lines.append(
                        "<details><summary>Show full merged table</summary>\n\n"
                    )
                    lines.extend(_render_value(var, rec["value"]))
                    lines.append("\n</details>\n\n")
                else:
                    lines.extend(_render_value(var, rec["value"]))
                    lines.append("\n\n")

    with mkdocs_gen_files.open("api/constants_reference.md", "w") as f:
        f.writelines(lines)


main()
