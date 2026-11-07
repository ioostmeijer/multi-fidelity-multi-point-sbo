import os

import numpy as np
import pytest
from logger_helper import _make_logger
from mf2 import bi_fidelity_functions
from smt.sampling_methods import LHS
from smt.surrogate_models import RBF as SMT_RBF
from surrogate_plotter import plot_1d, plot_2d

from sbo.surrogate import RBF
from sbo.utils import Config

# Comparison plots for a visual check. They are only saved when SBO_TEST_PLOT_DIR is
# set to a directory, e.g. SBO_TEST_PLOT_DIR=plots pytest -m verification; by default
# the tests write no files. The path is made absolute here, at import, because each
# test runs in its own temporary working directory (see tests/conftest.py).
PLOT_DIR = os.getenv("SBO_TEST_PLOT_DIR")
if PLOT_DIR:
    PLOT_DIR = os.path.abspath(PLOT_DIR)


@pytest.mark.verification
def test_validate_RBF() -> None:
    """Validate the RBF wrapper against the SMT implementation."""
    for function in bi_fidelity_functions:
        function_high = function.functions[0]
        config = Config(
            used_algorithms={"surrogate": "RBF"},
            lb=function.l_bound,
            ub=function.u_bound,
            functions=[function_high],
            settings={},
            logger=_make_logger(),
        )

        # Initialise RBF surrogates
        surrogate = RBF(config=config)
        smt_config = config.settings.get("surrogate", {})
        smt_config.pop("data_dir", None)
        smt_surrogate = SMT_RBF(**smt_config)

        # Setup LHS
        n_dim = config.n_dimensions
        num_init_samples = [3 * n_dim, 5 * n_dim, 10 * n_dim]
        lhs = LHS(xlimits=config.bounds, seed=config.get("random_seed", 42))

        # Perform tests
        for num_samples in num_init_samples:
            x_init = lhs(num_samples)
            y_init = np.array([function_high(x) for x in x_init])

            # Set training data for custom surrogate
            training_data = [
                {"coord": x_init[i], "value": y_init[i], "fid": 0}
                for i in range(len(x_init))
            ]
            surrogate.set_training_data(training_data)
            surrogate.train()

            # Set training data for SMT surrogate
            smt_surrogate.set_training_values(x_init, y_init)
            smt_surrogate.train()

            # Test predictions
            n_eval = max(1000, 10 * num_samples)
            eval_sampler = LHS(
                xlimits=config.bounds, seed=config.get("random_seed", 42)
            )
            x_eval = eval_sampler(n_eval)

            y_pred_custom = np.array(
                [surrogate.predict(np.atleast_2d(x)) for x in x_eval]
            )
            y_pred_smt = smt_surrogate.predict_values(x_eval)

            # Create the output directory if plotting and flatten and sort output
            if n_dim < 3:
                if PLOT_DIR:
                    output_dir = os.path.join(PLOT_DIR, "RBF")
                    os.makedirs(output_dir, exist_ok=True)

                    # Since LHS has a random order, sort the lists used for plotting
                    if n_dim == 1:
                        x1 = x_eval.flatten()
                        order = np.argsort(x1)
                        x_eval = x_eval[order]
                        y_pred_custom = y_pred_custom.flatten()[order]
                        y_pred_smt = y_pred_smt.flatten()[order]

                    elif n_dim == 2:
                        y_pred_custom = y_pred_custom.flatten()
                        y_pred_smt = y_pred_smt.flatten()

                        # build a grid for contour plotting
                        nx, ny = 100, 10
                        gx = np.linspace(config.lb[0], config.ub[0], nx)
                        gy = np.linspace(config.lb[1], config.ub[1], ny)
                        Xg, Yg = np.meshgrid(gx, gy)  # shape (ny, nx)

                        grid_points = np.column_stack(
                            (Xg.ravel(order="F"), Yg.ravel(order="F"))
                        )
                        d2 = (x_eval[:, 0][:, None] - grid_points[None, :, 0]) ** 2 + (
                            x_eval[:, 1][:, None] - grid_points[None, :, 1]
                        ) ** 2
                        nearest_idx = np.argmin(d2, axis=0)

                        y_pred_custom = y_pred_custom[nearest_idx].reshape(nx, ny).T
                        y_pred_smt = y_pred_smt[nearest_idx].reshape(nx, ny).T

                        x_eval = grid_points

            if n_dim == 1 and PLOT_DIR:
                lines = [
                    {
                        "x": x_eval.flatten(),
                        "y": y_pred_custom.flatten(),
                        "label": "Custom RBF Prediction",
                        "color": "blue",
                    },
                    {
                        "x": x_eval.flatten(),
                        "y": y_pred_smt.flatten(),
                        "label": "SMT RBF Prediction",
                        "color": "red",
                        "linestyle": "--",
                    },
                ]

                plot_1d(
                    lines=lines,
                    x_label="Input",
                    y_label="Predicted Output",
                    title=f"RBF Surrogate Predictions for {function.name} with "
                    f"{num_samples} samples",
                    output_path=os.path.join(
                        output_dir,
                        f"RBF_1D_{function.name}_{num_samples}_samples.png",
                    ),
                )

            elif n_dim == 2 and PLOT_DIR:
                lines = [
                    {"x": x_eval, "y": y_pred_smt, "title": "SMT RBF Predictions"},
                    {
                        "x": x_eval,
                        "y": y_pred_custom,
                        "title": "Custom RBF Predictions",
                    },
                    {
                        "x": x_eval,
                        "y": np.abs(y_pred_smt - y_pred_custom),
                        "title": "Absolute Difference",
                    },
                    {
                        "x": x_eval,
                        "y": np.abs(y_pred_smt - y_pred_custom)
                        / (np.abs(y_pred_smt) + 1e-8),
                        "title": "Relative Difference",
                    },
                ]

                plot_2d(
                    lines=lines,
                    title=f"RBF Surrogate Predictions for {function.name} with "
                    f"{num_samples} samples",
                    output_path=os.path.join(
                        output_dir,
                        f"RBF_2D_{function.name}_{num_samples}_samples.png",
                    ),
                )

            # Assert predictions are close
            y1 = np.asarray(y_pred_custom).squeeze().ravel()
            y2 = np.asarray(y_pred_smt).squeeze().ravel()
            np.testing.assert_allclose(
                y1,
                y2,
                err_msg=(
                    f"RBF predictions do not match SMT for function "
                    f"{function.name} with {num_samples} samples."
                ),
                atol=1e-5,
            )

        Config.destroy_instance()
