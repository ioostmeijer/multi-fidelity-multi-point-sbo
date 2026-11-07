"""Generate the Settings Reference from the *_DEFAULT_SETTINGS dicts.

Based on a claude.ai generated file. The type, default and constraints come from the
settings dicts; the descriptions come from the string block placed directly before each
dict (its "docstring"), which lists the keys as::

    Keys:
        - 'key': type, description (default: ..., min: ...)

The type and the "(default: ...)" part are left out of the description, since the table
shows them from the dict itself.
"""

import ast
import importlib
import inspect
import re
import sys
from collections import defaultdict
from pathlib import Path

import mkdocs_gen_files

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src"
PACKAGE = SRC / "sbo"
SETTINGS_ROOT = PACKAGE / "DefaultSettings"

sys.path.insert(0, str(SRC))


def _is_settings_var(name: str) -> bool:
    """Return True for a settings dict name (the general one is DEFAULT_SETTINGS)."""
    return name == "DEFAULT_SETTINGS" or name.endswith("_DEFAULT_SETTINGS")


def _class_names() -> dict[str, str]:
    """Return {CLASSNAME: ClassName} for the algorithm classes of the package."""
    names = {}
    for package in ("sbo.infill", "sbo.surrogate", "sbo.optimisation"):
        try:
            module = importlib.import_module(package)
        except Exception as e:  # pragma: no cover - only affects the titles
            print(f"[gen_settings_table] Could not import {package}: {e}")
            continue
        for name, _cls in inspect.getmembers(module, inspect.isclass):
            names[name.upper()] = name
    return names


CLASS_NAMES = _class_names()


def _display(var_name: str) -> str:
    """Return the section title of a settings dict, e.g. GEI, TwoStep, KRG."""
    if var_name == "DEFAULT_SETTINGS":
        return "General"
    prefix = var_name.replace("_DEFAULT_SETTINGS", "")
    return CLASS_NAMES.get(prefix, prefix.replace("_", " ").title())


def _format_value(value: object) -> str:
    """Return a readable repr of a default value (numpy scalars as plain numbers)."""
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        items = ", ".join(_format_value(v) for v in value)
        return f"[{items}]" if isinstance(value, list) else f"({items})"
    return repr(value)


def _module_name(py_file: Path) -> str:
    """Convert a file path to a dotted module name relative to SRC."""
    return ".".join(py_file.relative_to(SRC).with_suffix("").parts)


def _collect_modules() -> dict[str, object]:
    """Return {dotted_module_name: module} for every .py under DefaultSettings/."""
    modules = {}
    for py_file in sorted(SETTINGS_ROOT.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        mod_name = _module_name(py_file)
        try:
            modules[mod_name] = importlib.import_module(mod_name)
        except Exception as e:
            print(f"[gen_settings_table] Skipping {mod_name}: {e}")
    return modules


def _collect_settings(modules: dict) -> list[dict]:
    """Return a flat list of records, one per *_DEFAULT_SETTINGS variable.

    Each record:
        {
            "var_name":    "GEI_DEFAULT_SETTINGS",
            "display":     "GEI",               # var_name minus _DEFAULT_SETTINGS
            "module":      "sbo.DefaultSettings.infill.GEI",
            "category":    "infill",             # subfolder name
            "dict":        {...},                # the actual dict object
            "dict_id":     id(the_dict),
        }
    """
    records = []

    for mod_name, mod in modules.items():
        # Category = first subfolder under DefaultSettings/
        parts = mod_name.split(".")
        settings_idx = parts.index("DefaultSettings")
        category_parts = parts[settings_idx + 1 : -1]  # subfolders between root & file
        category = category_parts[0].capitalize() if category_parts else "General"

        for var_name, value in inspect.getmembers(mod):
            if not _is_settings_var(var_name):
                continue
            if not isinstance(value, dict):
                continue
            # Only collect variables *defined* in this module, not re-imported ones
            if not hasattr(mod, var_name):
                continue

            display = _display(var_name)

            records.append(
                {
                    "var_name": var_name,
                    "display": display,
                    "module": mod_name,
                    "category": category,
                    "dict": value,
                    "dict_id": id(value),
                }
            )

    return records


ITEM_RE = re.compile(r"^\s*-\s+'?([A-Za-z_]\w*)'?\s*:\s*(.*)$")


def _remove_default_groups(text: str) -> str:
    """Remove every "(default: ...)" group, including nested parentheses."""
    out, i = [], 0
    while i < len(text):
        if text.startswith("(default", i):
            depth = 0
            j = i
            while j < len(text):
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            i = j + 1
            continue
        out.append(text[i])
        i += 1
    return re.sub(r"\s+", " ", "".join(out)).strip().rstrip(".,") + "."


def _parse_key_descriptions(block: str) -> dict[str, str]:
    """Parse the "- 'key': type, description" items of a settings docstring block."""
    descriptions: dict[str, list[str]] = {}
    current = None
    item_indent = 0
    for line in block.splitlines():
        indent = len(line) - len(line.lstrip())
        match = ITEM_RE.match(line)
        if match:
            current = match.group(1)
            item_indent = indent
            descriptions[current] = [match.group(2)]
        elif current and line.strip() and indent > item_indent:
            descriptions[current].append(line.strip())
        else:
            # blank line, a new non-key item ("- All keys from ..."), or text at the
            # level of the items ends the current item
            current = None
    result = {}
    for key, parts in descriptions.items():
        text = " ".join(parts)
        # Drop the leading type ("int | None, ...")
        if ", " in text:
            text = text.split(", ", 1)[1]
        text = _remove_default_groups(text)
        result[key] = text[0].upper() + text[1:] if text else ""
    return result


def _collect_descriptions() -> dict[str, dict[str, str]]:
    """Return {var_name: {key: description}} from the block before each dict."""
    result: dict[str, dict[str, str]] = {}
    for py_file in sorted(SETTINGS_ROOT.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        last_string = None
        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                last_string = node.value.value
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and _is_settings_var(target.id):
                        result[target.id] = (
                            _parse_key_descriptions(last_string) if last_string else {}
                        )
            last_string = None
    return result


def _parse_assignments(py_file: Path) -> dict[str, str | None]:
    """Return dict for every *_DEFAULT_SETTINGS assignment in the file.

    Return dict has shape: {var_name: parent_var_name | None}

    Detects three patterns:
        X = Y                          → pure alias,    parent = Y
        X = {**Y}                      → copy of Y,     parent = Y
        X = {**Y, "extra_key": {...}}  → extension,     parent = Y
        X = {"key": {...}}             → standalone,    parent = None
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
            if not _is_settings_var(target.id):
                continue

            var = target.id
            val = node.value

            # Pattern 1: X = Y  (name alias)
            if isinstance(val, ast.Name) and val.id.endswith("_DEFAULT_SETTINGS"):
                result[var] = val.id

            # Pattern 2/3: X = {**Y, ...}  (dict with optional unpacking)
            elif isinstance(val, ast.Dict):
                parent = None
                for key in val.keys:
                    # A None key in ast.Dict represents **unpacking
                    if key is None:
                        continue
                # Check for **Y spread: represented as key=None, value=Name in ast
                for k, v in zip(val.keys, val.values):
                    if k is None and isinstance(v, ast.Name):
                        if v.id.endswith("_DEFAULT_SETTINGS"):
                            parent = v.id
                            break
                result[var] = parent

            else:
                result[var] = None

    return result


def _build_inheritance(records: list[dict]) -> dict[str, str | None]:
    """Return {var_name: parent_var_name | None} by running AST over every file."""
    inheritance = {}
    for py_file in sorted(SETTINGS_ROOT.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        assignments = _parse_assignments(py_file)
        inheritance.update(assignments)

    # For anything not found by AST, fall back to id() comparison:
    # if two records share the same dict object, the one defined later is an alias
    id_to_first_var: dict[int, str] = {}
    for rec in records:
        did = rec["dict_id"]
        if did not in id_to_first_var:
            id_to_first_var[did] = rec["var_name"]
        else:
            # This dict object was already seen — mark it as alias of first occurrence
            if rec["var_name"] not in inheritance:
                inheritance[rec["var_name"]] = id_to_first_var[did]

    return inheritance


def _topo_sort(records: list[dict], inheritance: dict[str, str | None]) -> list[dict]:
    """Return records ordered so every parent appears before its children."""
    var_to_rec = {r["var_name"]: r for r in records}
    visited, order = set(), []

    def visit(var_name: str) -> list | None:
        if var_name in visited:
            return
        visited.add(var_name)
        parent = inheritance.get(var_name)
        if parent and parent in var_to_rec:
            visit(parent)
        if var_name in var_to_rec:
            order.append(var_to_rec[var_name])

    for rec in records:
        visit(rec["var_name"])
    return order


def _type_str(t: type) -> str:
    """Convert a type object to a clean string representation for the table."""
    s = str(t).replace("<class '", "").replace("'>", "")
    return s


def _render_table(settings: dict, descriptions: dict[str, str]) -> list[str]:
    """Render the settings as a Markdown table, with their descriptions."""
    lines = [
        "| Key | Description | Type | Default | Constraints |\n",
        "|-----|-------------|------|---------|-------------|\n",
    ]
    for key, meta in sorted(settings.items()):
        description = descriptions.get(key, "")
        if not description:
            print(f"[gen_settings_table] No description for setting '{key}'")
        description = (description or "—").replace("|", "\\|")
        typ = _type_str(meta["type"]).replace("|", "\\|")
        default = f"`{_format_value(meta['default'])}`"
        constraints = []
        if "min" in meta and "max" in meta:
            constraints.append(f"`{meta['min']}` ≤ value ≤ `{meta['max']}`")
        elif "min" in meta:
            constraints.append(f"min: `{meta['min']}`")
        elif "max" in meta:
            constraints.append(f"max: `{meta['max']}`")
        if "allowed" in meta:
            vals = ", ".join(f"`{v}`" for v in meta["allowed"])
            constraints.append(f"one of: {vals}")
        c = "<br>".join(constraints) if constraints else "—"
        lines.append(f"| `{key}` | {description} | `{typ}` | {default} | {c} |\n")
    return lines


def _extra_keys(child: dict, parent: dict) -> dict:
    """Return the keys that are new in child or differ from parent."""
    return {k: v for k, v in child.items() if k not in parent or parent[k] != v}


def _descriptions_for(
    var: str, inheritance: dict, all_descriptions: dict
) -> dict[str, str]:
    """Return the descriptions of var, falling back to those of its parents."""
    chain = []
    while var:
        chain.append(var)
        var = inheritance.get(var)
    merged: dict[str, str] = {}
    for name in reversed(chain):
        merged.update({k: v for k, v in all_descriptions.get(name, {}).items() if v})
    return merged


def _anchor(display: str) -> str:
    return display.lower().replace(" ", "-")


def main() -> None:
    """Generate the settings docs by collecting all *_DEFAULT_SETTINGS dicts."""
    modules = _collect_modules()
    records = _collect_settings(modules)
    inheritance = _build_inheritance(records)
    sorted_recs = _topo_sort(records, inheritance)

    # Group by category, preserving topo order within each group
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in sorted_recs:
        groups[rec["category"]].append(rec)

    # Look up dicts by var_name for parent resolution
    var_to_dict = {r["var_name"]: r["dict"] for r in records}
    all_descriptions = _collect_descriptions()

    lines = [
        "# Settings Reference\n\n",
        "Auto-generated from all `*_DEFAULT_SETTINGS` variables in "
        "`DefaultSettings/`. Add a new file and it appears here automatically.\n\n",
        "The descriptions come from the `Keys:` block above each settings dict in "
        "the source.\n\n",
        "!!! tip\n",
        "    All settings are validated at construction time against the "
        "**Type**, **Default**, and **Constraints** shown. See "
        "[Settings and Stopping Criteria](../user/settings.md) for how to pass "
        "them.\n\n",
    ]

    for category, recs in groups.items():
        lines.append(f"## {category}\n\n")
        for rec in recs:
            var = rec["var_name"]
            display = rec["display"]
            settings = rec["dict"]
            parent_var = inheritance.get(var)

            lines.append(f"### {display}\n\n")
            descriptions = _descriptions_for(var, inheritance, all_descriptions)

            if parent_var is None:
                # Standalone — full table
                lines.extend(_render_table(settings, descriptions))
                lines.append("\n\n")

            else:
                parent_dict = var_to_dict.get(parent_var, {})
                parent_display = _display(parent_var)
                parent_anchor = _anchor(parent_display)
                extra = _extra_keys(settings, parent_dict)

                if id(settings) == id(parent_dict) or not extra:
                    lines.append(
                        f'!!! info "Alias"\n'
                        f"    **{display}** uses exactly the same settings as "
                        f"[{parent_display}](#{parent_anchor}). No additional keys.\n\n"
                    )
                else:
                    lines.append(
                        f'!!! info "Extends {parent_display}"\n'
                        f"    All settings from [{parent_display}](#{parent_anchor}) "
                        "are available. The following keys are **added or "
                        "changed**:\n\n"
                    )
                    lines.extend(_render_table(extra, descriptions))
                    lines.append("\n\n")

    with mkdocs_gen_files.open("api/settings_reference.md", "w") as f:
        f.writelines(lines)


main()
