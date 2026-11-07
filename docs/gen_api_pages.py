# docs/gen_api_pages.py
# Automatically generates one API reference page per module in src/
# Based on the mkdocstrings recipe: https://mkdocstrings.github.io/recipes/

from pathlib import Path

import mkdocs_gen_files

src = Path(__file__).parent.parent / "src"
nav = mkdocs_gen_files.Nav()

nav["Settings Reference"] = "settings_reference.md"
nav["Constants Reference"] = "constants_reference.md"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    # Skip __init__ and __main__ files from nav (they still get pages)
    if parts[-1] in ("__init__", "__main__"):
        parts = parts[:-1]
        if not parts:
            continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# `{parts[-1]}`\n\n")
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Write the nav for the API section
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
