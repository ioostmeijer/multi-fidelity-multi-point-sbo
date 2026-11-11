# Developer Guide

This guide is for working on the code itself. Start with the [Architecture](architecture.md) to see how the parts fit
together, then [Extending the Framework](extending/index.md) to add an algorithm, and [Testing](testing.md) for the test
suites.

## Development Environment

The environment is managed with [uv](https://docs.astral.sh/uv/). Install the full, locked development environment (the
same one CI uses, including the test, docs and lint tools) and the pre-commit hooks:

```bash
uv sync --locked                            # creates .venv with the Python version in .python-version and all dev dependencies
uv run pre-commit install
```

Prefix commands with `uv run` (e.g. `uv run pytest`), or activate `.venv` first.

## Checks and Tests

```bash
pre-commit run --all-files                  # format + lint (see the hooks below)
deptry src                                  # dependency hygiene (unused / undeclared)

pytest -m unit                              # fast unit tests
pytest -m verification                      # slower verification tests
pytest -m e2e                               # end-to-end tests
mpiexec -n 4 python -m pytest --with-mpi -p no:xdist -m "mpi or mock_mpi"   # MPI tests
```

See [Testing](testing.md) for what each level covers.

### Pre-commit Hooks

`.pre-commit-config.yaml` defines the hooks, which run on every `git commit` (after `pre-commit install`) and in the
Lint workflow. They are `local` hooks that run the tools installed from the `lint` dependency group in `pyproject.toml`
(`uv run --no-sync <tool>`), so pre-commit doesn't install its own copies and the tool versions are set in one place
only: `uv.lock`. Upgrade them like any other dependency (see [Dependencies](#dependencies)); don't run
`pre-commit autoupdate`. The hooks need the `lint` group installed: `uv sync --locked` installs it, and
`uv sync --locked --only-group lint` installs only the hook tools.

| Hook                                                   | Files          | What it does                                                                                                     |
| ------------------------------------------------------ | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `ruff-check`                                           | Python         | Lint (pycodestyle, pyflakes, bugbear, builtins, comprehensions, annotations, NumPy docstrings) and sort imports. |
| `ruff-format`                                          | Python         | Format (line length 88).                                                                                         |
| `mdformat` (with `mdformat-mkdocs`, `mdformat-tables`) | Markdown       | Format: wrap at 120 characters, number lists, align tables; keeps the MkDocs syntax.                             |
| `codespell`                                            | All text files | Report common typos and American spellings (British spelling); see below.                                        |
| `end-of-file-fixer`, `trailing-whitespace`             | All text files | One final newline, no trailing spaces.                                                                           |

The formatting hooks change the files and fail the commit; add the changed files and commit again.

The project uses British spelling (optimisation, modelling, colour). `codespell` checks it with its default dictionaries
plus `.codespell-en-GB.txt`, which maps American spellings to British ones (configured under `[tool.codespell]` in
`pyproject.toml`). It only reports; fix the words by hand or with `uv run codespell --write-changes`. Names from other
packages (`scipy.optimize`, `color=`) and the titles of cited papers keep their American spelling: the `ignore-regex` in
`pyproject.toml` lists them. Add a new one there, or put `codespell:ignore <word>` in a comment on that line.

## Continuous Integration

The GitHub Actions workflows are in `.github/workflows/`. The test and docs workflows install the locked environment
from `uv.lock` (via the shared `.github/actions/setup-env` action); the Lint workflow installs only the locked `lint`
group (`uv sync --locked --only-group lint`), without the package or MPI, and runs the hooks with it.

| Workflow             | Runs on                                                    | Jobs                                                                  | Required check          |
| -------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------- |
| `lint.yml`           | All pushes and pull requests to `main` and `dev`           | Pre-commit hooks on all files                                         | `pre-commit`            |
| `ci.yml`             | All pushes and pull requests to `main` and `dev`           | deptry, unit tests (Linux and Windows)                                | `CI passed`             |
| `extended.yml`       | Pull requests and pushes to `main`; manually on any branch | Verification and E2E tests; coverage (manual)                         | `Extended tests passed` |
| `docs.yml`           | Pushes to `main` and all pull requests                     | Builds the docs; deploys them on `main`                               | `build`                 |
| `project-in-dev.yml` | Merged pull requests into `dev`                            | Moves the linked issues (`Part of #N`) to In dev on the project board | -                       |

`ci.yml` and `extended.yml` start with a `changes` job (the shared `.github/actions/detect-code-changes` action), which
checks whether `src/`, `tests/`, `pyproject.toml`, `uv.lock`, the shared actions or the workflow itself changed. If not,
for example in a documentation-only pull request, the tests are skipped. Each of these workflows ends with a summary job
(`CI passed`, `Extended tests passed`) that fails if any of its jobs failed or was cancelled, and passes if they passed
or were skipped. Branch protection on `main` requires only the summary jobs, `pre-commit` and `build`, so a
documentation-only pull request can be merged within a minute, while a pull request that changes code has to pass every
test.

The extended tests are slow, so they run on a pull request to `main` only once it is out of draft. Open the pull request
as a draft while you are still pushing to it, then mark it **Ready for review** to run them; a later push cancels the
run in progress and starts a new one. They aren't run again on the merge commit. To run them on another branch, open
**Actions → Extended tests → Run workflow** and select the branch, or run `gh workflow run extended.yml --ref <branch>`.

## Dependencies

Dependencies are declared in `pyproject.toml`:

- runtime dependencies under `[project] dependencies`;
- the `benchmarks` extra under `[project.optional-dependencies]`: deprecated and empty since the benchmark problems are
    part of the package (`sbo.benchmarks`), kept until version 2.0.0 so existing install commands keep working;
- the development-only `test`, `docs`, `lint` and `dev` groups under `[dependency-groups]` (`dev` includes `test`,
    `docs` and `lint`; `lint` holds the pre-commit hook tools).

The exact versions of everything, including transitive dependencies, are pinned in `uv.lock`. Never edit `uv.lock` by
hand.

```bash
uv add <package>                            # add a runtime dependency
uv add --optional benchmarks <package>      # add a dependency to the benchmarks extra
uv add --group test <package>               # add a development dependency to a group
uv lock --upgrade                           # upgrade all pins within the constraints in pyproject.toml
uv lock --upgrade-package black             # upgrade one package, e.g. a hook tool
```

Commit `pyproject.toml` and `uv.lock` together; CI installs with `uv sync --locked`, which fails if they are out of
sync.

!!! warning "Licences of new dependencies"

    The package is MIT-licensed. A new runtime dependency must have a licence that allows this (MIT, BSD, Apache-2.0, PSF,
    ...). A GPL-licensed package can only be an optional extra.

`[tool.uv] constraint-dependencies` in `pyproject.toml` keeps `pymdown-extensions` below version 12, which is
incompatible with the mkdocstrings version used for the API reference. `deptry` ignores `setuptools` for the "unused
dependency" rule: nothing imports it, and its `<81` pin only keeps an installed mf2 working (mf2 imports
`pkg_resources`) while `"module": "mf2"` is still supported in `Config.from_json`. Both go in version 2.0.0.

To export the lock to a `requirements.txt` (e.g. for environments without uv):

```bash
uv export --locked --no-dev --no-hashes -o requirements.txt
```

## Documentation

The docs are built with MkDocs (Material theme). Build them, and catch any broken references, with:

```bash
uv run mkdocs build --strict     # uv run mkdocs serve for a live preview
```

The Docs workflow runs the same strict build, so make sure it passes before opening a pull request.

**Generated automatically (no manual edits needed):**

- **API Reference:** every module under `src/sbo/` gets its own page, generated by `docs/gen_api_pages.py` from the
    docstrings. Write the docstrings in [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html)
    (`Parameters`, `Returns`, `Raises`, `Notes` and `Examples` sections, each underlined with dashes; summary in the
    imperative mood). A `Returns` entry starts with the type on its own line, followed by the indented description:

    ```python
    def calc_infill_value(self, x: np.ndarray, surrogate: SurrogateBase) -> float:
        """Calculate the infill value at x.

        Parameters
        ----------
        x
            The point at which to calculate the infill value.
        surrogate
            The surrogate model used in the calculation.

        Returns
        -------
        float
            The infill value at x.
        """
    ```

    Types do not have to be repeated in the `Parameters` section, as they are taken from the type hints in the signature.

    **Formulas** are written in LaTeX between dollar signs: `$...$` inline, and `$$` on the lines before and after a
    formula block (with a blank line around it). `pymdownx.arithmatex` and MathJax (see `mkdocs.yml`) typeset them in
    the docs; the same syntax works in the Markdown pages. Make a docstring with LaTeX backslashes a raw string
    (`r"""`), so Python doesn't read `\s` in `\sin` as an escape. Don't use the reStructuredText syntax `.. math::` or
    `:math:`, which is meant for Sphinx: MkDocs shows it as plain text.

    ```python
    def forrester_high(x: np.ndarray) -> np.ndarray:
        r"""Evaluate the high fidelity of the Forrester function.

        $$
        f_h(x) = (6x - 2)^2 \sin(12x - 4)
        $$
        """
    ```

- **Settings Reference:** every `*_DEFAULT_SETTINGS` dict in `src/sbo/DefaultSettings/` is listed by
    `docs/gen_settings_table.py`, with its type, default and constraints from the dict, and the description from the
    `Keys:` block above the dict (see [Default Settings](extending/index.md#default-settings)).

- **Constants Reference:** every public constant in `src/sbo/constants/` is listed by `docs/gen_constants_table.py`.

- **Examples:** the example scripts are included in the user pages with snippets (`--8<-- "examples/<script>.py"`), so
    they always match the code. The build fails if an included file does not exist.

**Update by hand:**

- **Algorithm tables:** add a new algorithm to the tables on the docs home page (`docs/index.md`) and in `README.md`,
    with its status.
- **User pages:** if the new algorithm needs a different setup, add or update the page of its topic, and add an example
    script to `examples/` if useful (and list it in [Getting Started](../user/getting_started.md#examples)).
- **References:** if the implementation is based on a paper, add it to the `References` section of the class docstring
    and to the references in `docs/index.md` and `README.md`.

Markdown files are formatted by the `mdformat` pre-commit hook (wrapped at 120 characters, aligned tables); run
`uv run mdformat --wrap 120 --number <file>` to format a file by hand.

Diagrams follow the colours of the draw.io diagrams in `docs/assets/`: per class, `Config` purple, `SBO` blue, the
optimisation algorithm green, the surrogate orange, the infill strategy yellow, and the user, inputs and outputs grey.
In a Mermaid diagram, give a node one of the classes `config`, `sbo`, `optimisation`, `surrogate`, `infill` or `user`
(`classDef sbo stroke-width:1px` and `class MyNode sbo`); `docs/javascripts/mermaid_style.js` colours it, in light and
dark mode, with the colours defined in `docs/stylesheets/extra.css`.

## Running on HPC

The Apptainer definition and build script are in the `HPC/` folder; see [Running on HPC](../user/hpc.md).
