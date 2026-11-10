"""The building blocks of the benchmark problems.

Every benchmark problem is one :class:`MultiFidelityFunction` instance, holding a plain
function per fidelity, the bounds of the design space and the known optimum.
:class:`FunctionSet` describes the interface that ``Config.from_json`` needs, so users
can provide their own problems without inheriting from anything.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

FidelityFunction = Callable[[np.ndarray], np.ndarray]
"""One fidelity: maps points with shape (n, n_dim) to values with shape (n,)."""


@runtime_checkable
class FunctionSet(Protocol):
    """The interface of a set of functions that ``Config.from_json`` can use.

    Any object with these attributes works, for example a :class:`MultiFidelityFunction`
    or an instance of a user's own class.
    """

    @property
    def functions(self) -> Sequence[FidelityFunction]:
        """The functions per fidelity, from high to low fidelity."""
        ...

    @property
    def l_bound(self) -> np.ndarray:
        """The lower bound of every design variable, with shape (n_dim,)."""
        ...

    @property
    def u_bound(self) -> np.ndarray:
        """The upper bound of every design variable, with shape (n_dim,)."""
        ...


def as_2d(x: np.ndarray, n_dim: int) -> np.ndarray:
    """Convert the input of a fidelity function to a 2D array of points.

    Parameters
    ----------
    x
        One point with shape (n_dim,), or several points with shape (n, n_dim). For a
        1D problem, several points must have shape (n, 1): shape (n,) is read as one
        point with n dimensions.
    n_dim
        The number of dimensions of the problem.

    Returns
    -------
    np.ndarray
        The points as a float array with shape (n, n_dim).

    Raises
    ------
    ValueError
        If the points don't have n_dim dimensions.
    """
    points = np.atleast_2d(np.asarray(x, dtype=float))
    if points.ndim != 2 or points.shape[1] != n_dim:
        raise ValueError(
            f"Expected points with shape (n, {n_dim}) or ({n_dim},), "
            f"got shape {np.shape(x)}."
        )
    return points


def _read_only_vector(values: np.ndarray, field_name: str) -> np.ndarray:
    """Return a read-only 1D float copy of ``values``.

    Parameters
    ----------
    values
        The values to convert.
    field_name
        The name of the field, used in the error message.

    Returns
    -------
    np.ndarray
        A read-only 1D float array.

    Raises
    ------
    ValueError
        If the values are not 1D or contain NaN or infinite values.
    """
    vector = np.array(np.atleast_1d(values), dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{field_name} must be 1D, got shape {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} must be finite, got {vector}.")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, eq=False)
class MultiFidelityFunction:
    """A benchmark problem with one function per fidelity.

    The instances are read-only (``frozen=True``), and so are the arrays they hold, so
    a benchmark that is shared by the whole package can't be changed by accident.
    Instances compare by identity (``eq=False``), because comparing NumPy arrays
    field by field is ambiguous.

    Attributes
    ----------
    name
        Lower-case name without spaces, e.g. ``"six_hump_camelback"``. Used by
        ``get_benchmark`` and as the key of the function delays.
    functions
        One function per fidelity, from high to low fidelity. Each maps points with
        shape (n, n_dim) or (n_dim,) to values with shape (n,).
    l_bound
        The lower bound of every design variable, with shape (n_dim,).
    u_bound
        The upper bound of every design variable, with shape (n_dim,).
    x_opt
        The location of the known optimum of the highest fidelity, with shape
        (n_dim,), or None if unknown.
    f_opt
        The value of the known optimum of the highest fidelity, or None if unknown.
    """

    name: str
    functions: tuple[FidelityFunction, ...]
    l_bound: np.ndarray
    u_bound: np.ndarray
    x_opt: np.ndarray | None = None
    f_opt: float | None = None

    def __post_init__(self) -> None:
        """Check the fields and convert them to read-only types.

        Raises
        ------
        ValueError
            If the name is invalid, there are no functions, the bounds don't match, or
            the optimum is outside the bounds.
        TypeError
            If one of the functions is not callable.
        """
        # A frozen dataclass blocks normal assignment, so object.__setattr__ is used
        # to store the converted values.
        if not self.name or self.name != self.name.lower() or " " in self.name:
            raise ValueError(
                f"name must be lower case without spaces, got '{self.name}'."
            )

        functions = tuple(self.functions)
        if not functions:
            raise ValueError("functions must contain at least one fidelity.")
        for function in functions:
            if not callable(function):
                raise TypeError(f"Every fidelity must be callable, got {function!r}.")
        object.__setattr__(self, "functions", functions)

        l_bound = _read_only_vector(self.l_bound, "l_bound")
        u_bound = _read_only_vector(self.u_bound, "u_bound")
        if l_bound.shape != u_bound.shape:
            raise ValueError(
                f"l_bound and u_bound must have the same shape, got {l_bound.shape} "
                f"and {u_bound.shape}."
            )
        if not np.all(l_bound < u_bound):
            raise ValueError(
                f"l_bound must be smaller than u_bound, got {l_bound} and {u_bound}."
            )
        object.__setattr__(self, "l_bound", l_bound)
        object.__setattr__(self, "u_bound", u_bound)

        if self.x_opt is not None:
            x_opt = _read_only_vector(self.x_opt, "x_opt")
            if x_opt.shape != l_bound.shape:
                raise ValueError(
                    f"x_opt must have shape {l_bound.shape}, got {x_opt.shape}."
                )
            if not np.all((l_bound <= x_opt) & (x_opt <= u_bound)):
                raise ValueError(f"x_opt {x_opt} is outside the bounds.")
            object.__setattr__(self, "x_opt", x_opt)

        if self.f_opt is not None:
            object.__setattr__(self, "f_opt", float(self.f_opt))

    @property
    def n_dim(self) -> int:
        """int: The number of design variables."""
        return self.l_bound.shape[0]

    @property
    def n_fidelities(self) -> int:
        """int: The number of fidelities."""
        return len(self.functions)
