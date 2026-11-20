"""Shared types for the points and samples.

The types are ``TypedDict``s: at runtime a point is a plain ``dict``, so
``point["coord"]`` keeps working, while type checkers (mypy, your editor) check the key
names and value types. Fidelity 0 is the highest fidelity (HF).

Examples
--------
>>> import numpy as np
>>> from sbo.types import EvaluatedPoint
>>> sample: EvaluatedPoint = {"coord": np.array([0.5, 1.0]), "fid": 0, "value": 3.2}
"""

from typing import NotRequired, TypeAlias, TypedDict

import numpy as np


class Point(TypedDict):
    """A location in the design space at a fidelity level, not evaluated yet.

    Returned by the infill strategies as the next points to evaluate.

    Attributes
    ----------
    coord
        The coordinates of the point, with shape (n_dimensions,).
    fid
        The fidelity level (0 = HF, higher numbers are lower fidelities).
    start_time
        The time the evaluation was requested or started (``time.time()``). Only set
        by the optimisation, e.g. when the point is sent to an MPI worker.
    """

    coord: np.ndarray
    fid: int
    start_time: NotRequired[float]


class EvaluatedPoint(Point):
    """A point with its function value: a sample of the surrogate.

    Also used for the optimum of a surrogate (the value is then the prediction) and for
    the result of the optimisation.

    Attributes
    ----------
    value
        The function value at the point (or the predicted value for a surrogate
        optimum).
    mpi_rank
        The MPI rank that evaluated the point. Only set by the optimisation.
    end_time
        The time the evaluation ended (``time.time()``). Only set by the optimisation.
    """

    value: float
    mpi_rank: NotRequired[int]
    end_time: NotRequired[float]


Task: TypeAlias = list[Point]
"""Points that are evaluated together, e.g. on one MPI worker."""

Samples: TypeAlias = dict[int, list[EvaluatedPoint]]
"""The samples per fidelity level: ``{fid: [EvaluatedPoint, ...], ...}``."""

FunctionCalls: TypeAlias = dict[int, int]
"""The number of function calls per fidelity level: ``{fid: n_calls, ...}``."""

InfillResult: TypeAlias = tuple[list[Task], float]
"""The new tasks to evaluate and the infill value of the selected point."""

__all__ = [
    "EvaluatedPoint",
    "FunctionCalls",
    "InfillResult",
    "Point",
    "Samples",
    "Task",
]
