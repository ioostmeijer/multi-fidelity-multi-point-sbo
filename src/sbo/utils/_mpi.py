"""The optional ``mpi4py.MPI`` module, or ``None`` if mpi4py isn't available.

mpi4py is a dependency, but it fails to load without an MPI library (e.g. on Windows
without MS-MPI); the package then runs without MPI. Import ``MPI`` from here instead of
from mpi4py, and check ``MPI is not None`` before using it. The tests replace the
``MPI`` name in the module that uses it, so import the name (``from ..utils._mpi import
MPI``), not this module.
"""

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    # Typed as the mpi4py module, so the code that uses it keeps its types
    MPI = None  # type: ignore[assignment]

__all__ = ["MPI"]
