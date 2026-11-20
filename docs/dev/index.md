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
uv audit --locked                           # known vulnerabilities in the locked dependencies (uv.lock)

pytest -m unit                              # fast unit tests
pytest -m verification                      # slower verification tests
pytest -m e2e                               # end-to-end tests
mpiexec -n 4 python -m pytest --with-mpi -p no:xdist -m "mpi or mock_mpi"   # MPI tests
```

The same checks run in CI through the [nox sessions](testing.md#nox-sessions) in `noxfile.py`, each in its own
environment with the locked versions from `uv.lock`, and with the tests on every supported Python version:

```bash
uv run --only-group nox nox                 # lint, deptry, docs, and the unit tests on Python 3.11 and 3.14
uv run --only-group nox nox -s tests-3.11   # one session; `nox --list` shows them all
```

See [Testing](testing.md) for what each level covers.

### Pre-commit Hooks

`.pre-commit-config.yaml` defines the hooks, which run on every `git commit` (after `pre-commit install`) and in the
Lint workflow. They are `local` hooks that run the tools installed from the `lint` dependency group in `pyproject.toml`
(`uv run --no-sync <tool>`), so pre-commit doesn't install its own copies and the tool versions are set in one place
only: `uv.lock`. Upgrade them like any other dependency (see [Dependencies](#dependencies)); don't run
`pre-commit autoupdate`. The hooks need the `lint` group installed, and mypy also needs the dependencies of the package
for their types: `uv sync --locked` installs both, and
`uv sync --locked --no-install-project --no-default-groups --group lint` installs only these (as the `lint` nox session
does).

| Hook                                                   | Files          | What it does                                                                                                             |
| ------------------------------------------------------ | -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `ruff-check`                                           | Python         | Lint (pycodestyle, pyflakes, bugbear, builtins, comprehensions, annotations, NumPy docstrings, bandit) and sort imports. |
| `ruff-format`                                          | Python         | Format (line length 88).                                                                                                 |
| `mypy`                                                 | Python         | Type check the package (`src/sbo`); see below.                                                                           |
| `mdformat` (with `mdformat-mkdocs`, `mdformat-tables`) | Markdown       | Format: wrap at 120 characters, number lists, align tables; keeps the MkDocs syntax.                                     |
| `codespell`                                            | All text files | Report common typos and American spellings (British spelling); see below.                                                |
| `end-of-file-fixer`, `trailing-whitespace`             | All text files | One final newline, no trailing spaces.                                                                                   |

The formatting hooks change the files and fail the commit; add the changed files and commit again.

### Type Checking

mypy checks the type hints of the package (`src/sbo`), configured under `[tool.mypy]` in `pyproject.toml`. The hook
always checks the whole package (`uv run mypy`), not only the changed files. All dependencies have type hints: numpy,
matplotlib, smt and mpi4py ship them, and scipy gets them from `scipy-stubs` (in the `lint` group), whose version
follows scipy's. The whole package passes, so new code has to pass too. When fixing an error, prefer correcting the code
or the annotation, or narrowing the type (e.g. `if x is None: raise ...`), over `typing.cast`. Use
`# type: ignore[<code>]` only as a last resort, with a comment saying why; `warn_unused_ignores` reports it once it is
no longer needed. On a signature that spans several lines, mypy reports the error on the `def` line, so the comment goes
there.

Points and samples are plain dicts typed with the `TypedDict`s and aliases in `src/sbo/types.py` (`Point`,
`EvaluatedPoint`, `Task`, `Samples`, `FunctionCalls`, `InfillResult`); use them instead of `dict` or `list[dict]` in new
code, see [Architecture](architecture.md#data-structures). When building a point, annotate the variable
(`task: Task = [{"coord": x, "fid": 0}]`), so mypy checks the keys of the literal. A list of samples
(`list[EvaluatedPoint]`) is not a `list[Point]` for mypy (lists are invariant): take `Sequence[Point]` in a parameter
that only reads the points. In 2.0.0 these types become dataclasses (#61).

The package ships an empty `src/sbo/py.typed` ([PEP 561](https://peps.python.org/pep-0561/)), so mypy and other type
checkers use its type hints in the projects that install it.

The project uses British spelling (optimisation, modelling, colour). `codespell` checks it with its default dictionaries
plus `.codespell-en-GB.txt`, which maps American spellings to British ones (configured under `[tool.codespell]` in
`pyproject.toml`). It only reports; fix the words by hand or with `uv run codespell --write-changes`. Names from other
packages (`scipy.optimize`, `color=`) and the titles of cited papers keep their American spelling: the `ignore-regex` in
`pyproject.toml` lists them. Add a new one there, or put `codespell:ignore <word>` in a comment on that line.

## Continuous Integration

The GitHub Actions workflows are in `.github/workflows/`. Their steps run the [nox sessions](testing.md#nox-sessions)
(`uv run --locked --only-group nox nox -s <session>`), so the commands are defined in one place, `noxfile.py`, and run
the same way locally. Each session installs the locked dependencies of its group from `uv.lock`; the `lint` session
installs only the hook tools, without the package or MPI. The test and docs workflows first run the shared
`.github/actions/setup-env` action, which installs uv, Python, MPI (Linux) and nox; the OS-specific setup, such as MPI
and the Windows long-paths fix, stays in the workflows. Each test job keeps its own operating system and Python version
from the matrix and runs one session, e.g. `nox -s "tests-${{ matrix.python-version }}"`.

| Workflow             | Runs on                                                    | Jobs                                                                  | Required check          |
| -------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------- |
| `lint.yml`           | All pushes and pull requests to `main` and `dev`           | Pre-commit hooks on all files (incl. mypy)                            | `pre-commit`            |
| `ci.yml`             | All pushes and pull requests to `main` and `dev`           | deptry, unit tests (Linux and Windows)                                | `CI passed`             |
| `extended.yml`       | Pull requests and pushes to `main`; manually on any branch | Verification and E2E tests; coverage (manual)                         | `Extended tests passed` |
| `docs.yml`           | Pushes to `main` and all pull requests                     | Builds the docs; deploys them on `main`                               | `build`                 |
| `audit.yml`          | Dependency changes to `main` and `dev`; weekly             | `uv audit` on the locked dependencies (all groups)                    | -                       |
| `project-in-dev.yml` | Merged pull requests into `dev`                            | Moves the linked issues (`Part of #N`) to In dev on the project board | -                       |
| `release.yml`        | Pushes to `main`; manually on any branch (pre-release)     | Builds the wheel and sdist and creates the GitHub Release             | -                       |

`ci.yml` and `extended.yml` start with a `changes` job (the shared `.github/actions/detect-code-changes` action), which
checks whether `src/`, `tests/`, `pyproject.toml`, `uv.lock`, `noxfile.py`, the shared actions or the workflow itself
changed. If not, for example in a documentation-only pull request, the tests are skipped. Each of these workflows ends
with a summary job (`CI passed`, `Extended tests passed`) that fails if any of its jobs failed or was cancelled, and
passes if they passed or were skipped. Branch protection on `main` requires only the summary jobs, `pre-commit` and
`build`, so a documentation-only pull request can be merged within a minute, while a pull request that changes code has
to pass every test.

The extended tests are slow, so they run on a pull request to `main` only once it is out of draft. Open the pull request
as a draft while you are still pushing to it, then mark it **Ready for review** to run them; a later push cancels the
run in progress and starts a new one. They aren't run again on the merge commit. To run them on another branch, open
**Actions → Extended tests → Run workflow** and select the branch, or run `gh workflow run extended.yml --ref <branch>`.

## Releasing

A release is made by merging `dev` into `main` with a new version; `release.yml` then creates the tag and the GitHub
Release. Don't create release tags by hand. To prepare a release, in a pull request into `dev` (or the release pull
request into `main`):

1. Set the new version in `pyproject.toml` (`X.Y.Z`, see [Semantic Versioning](https://semver.org/spec/v2.0.0.html)).
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a new empty `## [Unreleased]` above it,
    and update the links at the bottom (`[X.Y.Z]` and the `[unreleased]` compare link).

When the pull request into `main` is merged (after the required checks passed), `release.yml`:

- reads the version from `pyproject.toml`; if the tag `vX.Y.Z` already exists, nothing is released (e.g. a merge without
    a new version);
- builds the sdist and the wheel with `uv build`, and checks the wheel in a clean environment (installs, imports, has
    the right version and `py.typed`);
- creates the tag `vX.Y.Z` and the GitHub Release, with the `## [X.Y.Z]` section of `CHANGELOG.md` as its text and the
    wheel and sdist attached. It fails if the section is missing or empty, or if the release already exists.

The tests are not run again: `main` only accepts pull requests that passed them. The package is not published on PyPI.

### Testing a Release

A pre-release version ([PEP 440](https://peps.python.org/pep-0440/): `1.2.0.dev1`, `1.2.0a1`, `1.2.0b1`, `1.2.0rc1`)
comes before the final `1.2.0`; pip skips it unless asked (`--pre`), and GitHub labels it **Pre-release** and doesn't
mark it as the latest release. Use one to test the release workflow without making a real release. Outside `main`,
`release.yml` only accepts a pre-release version (and on `main` only a final version), and uses the `## [Unreleased]`
section as its text.

1. On a throwaway branch, set the version in `pyproject.toml` to a pre-release, e.g. `1.2.0rc1`.
2. Run the workflow on that branch: **Actions → Release → Run workflow**, or
    `gh workflow run release.yml --ref <branch>`. A manual run needs `release.yml` on `main`; to test a change to the
    workflow itself, also add the branch under `on: push: branches:` in `release.yml` on that branch, and push.
3. Check the pre-release and its files, then remove it with `gh release delete v1.2.0rc1 --cleanup-tag`, and delete the
    branch.

## Dependencies

Dependencies are declared in `pyproject.toml`:

- runtime dependencies under `[project] dependencies`;
- the `benchmarks` extra under `[project.optional-dependencies]`: deprecated and empty since the benchmark problems are
    part of the package (`sbo.benchmarks`), kept until version 2.0.0 so existing install commands keep working;
- the development-only `test`, `docs`, `lint`, `dev` and `nox` groups under `[dependency-groups]` (`dev` includes
    `test`, `docs` and `lint`; `lint` holds the pre-commit hook tools; `nox` holds only nox, which runs the sessions in
    `noxfile.py`).

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
`pkg_resources`) while `"module": "mf2"` is still supported in `Config.from_json`. Because of that pin, the `audit`
session in `noxfile.py` (run by `audit.yml`) ignores two setuptools advisories that are fixed only in later versions.
All three go in version 2.0.0.

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
