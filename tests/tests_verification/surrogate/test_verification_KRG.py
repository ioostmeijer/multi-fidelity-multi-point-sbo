import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from logger_helper import _make_logger
from mf2 import bi_fidelity_functions
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG as SMT_KRG

from sbo.surrogate import KRG
from sbo.utils import Config

# Comparison plots for a visual check. They are only saved when SBO_TEST_PLOT_DIR is
# set to a directory, e.g. SBO_TEST_PLOT_DIR=plots pytest -m verification; by default
# the tests write no files. The path is made absolute here, at import, because each
# test runs in its own temporary working directory (see tests/conftest.py).
PLOT_DIR = os.getenv("SBO_TEST_PLOT_DIR")
if PLOT_DIR:
    PLOT_DIR = os.path.abspath(PLOT_DIR)


@pytest.mark.verification
def test_validate_KRG() -> None:
    """Validate the KRG surrogate model against the SMT implementation."""
    for function in bi_fidelity_functions:
        function_high = function.functions[0]
        config = Config(
            used_algorithms={"surrogate": "KRG"},
            lb=function.l_bound,
            ub=function.u_bound,
            functions=[function_high],
            settings={},
            logger=_make_logger(),
        )

        # Initialise KRG surrogates
        surrogate = KRG(config=config)
        smt_config = config.settings.get("surrogate", {})
        smt_config.pop("data_dir", None)
        smt_surrogate = SMT_KRG(**smt_config)

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

            y_pred_custom = surrogate.predict_values(x_eval)
            y_pred_smt = smt_surrogate.predict_values(x_eval)

            # Test variances
            y_var_custom = surrogate.predict_variance(x_eval)
            y_var_smt = smt_surrogate.predict_variances(x_eval)

            # Create the output dir if plotting and flatten and sort output
            if n_dim < 3:
                if PLOT_DIR:
                    output_dir = os.path.join(PLOT_DIR, "KRG")
                    os.makedirs(output_dir, exist_ok=True)

                # Since LHS has a random order, sort the lists used for plotting
                if n_dim == 1:
                    x1 = x_eval.flatten()
                    order = np.argsort(x1)
                    x_eval = x_eval[order]
                    y_pred_custom = y_pred_custom.flatten()[order]
                    y_pred_smt = y_pred_smt.flatten()[order]
                    y_var_custom = y_var_custom.flatten()[order]
                    y_var_smt = y_var_smt.flatten()[order]

                elif n_dim == 2:
                    y_pred_custom = y_pred_custom.flatten()
                    y_pred_smt = y_pred_smt.flatten()
                    y_var_custom = y_var_custom.flatten()
                    y_var_smt = y_var_smt.flatten()

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
                    y_var_custom = y_var_custom[nearest_idx].reshape(nx, ny).T
                    y_var_smt = y_var_smt[nearest_idx].reshape(nx, ny).T

                    x_eval = grid_points

            # Plot results if 1D
            if n_dim == 1 and PLOT_DIR:
                plt.figure()
                plt.plot(
                    x_eval,
                    y_pred_custom,
                    label="Custom KRG Prediction",
                    color="blue",
                )
                plt.fill_between(
                    x_eval.flatten(),
                    y_pred_custom - 1.96 * np.sqrt(y_var_custom),
                    y_pred_custom + 1.96 * np.sqrt(y_var_custom),
                    color="blue",
                    alpha=0.2,
                    label="Custom KRG 95% CI",
                )
                plt.plot(
                    x_eval,
                    y_pred_smt,
                    label="SMT KRG Prediction",
                    color="orange",
                    linestyle="--",
                )
                plt.fill_between(
                    x_eval.flatten(),
                    y_pred_smt - 1.96 * np.sqrt(y_var_smt),
                    y_pred_smt + 1.96 * np.sqrt(y_var_smt),
                    color="orange",
                    alpha=0.2,
                    label="SMT KRG 95% CI",
                )
                plt.scatter(x_init, y_init, color="black", label="Training Data")
                plt.title(f"KRG Surrogate Model ({num_samples} samples)")
                plt.xlabel("Input")
                plt.ylabel("Output")
                plt.legend(loc="upper left")
                plt.grid()

                # Plot the difference on second axis
                ax2 = plt.gca().twinx()
                y_pred_diff = y_pred_custom - y_pred_smt
                y_var_diff = y_var_custom - y_var_smt
                ax2.plot(
                    x_eval.flatten(),
                    y_pred_diff,
                    label="Prediction difference",
                    c="tab:purple",
                )
                ax2.plot(
                    x_eval.flatten(),
                    y_var_diff,
                    label="Variance difference",
                    linestyle="--",
                    c="tab:olive",
                )
                ax2.legend(loc="upper right")
                ax2.set_xlabel("Custom-SMT")
                ax2.set_ylim(
                    [
                        min(min(y_var_diff), min(y_pred_diff)) - 1e-6,
                        max(max(y_var_diff), max(y_pred_diff)) + 1e-6,
                    ]
                )
                plt.xlim(config.lb[0], config.ub[0])
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        output_dir,
                        f"KRG_validation_{function.name}_{num_samples}_samples.png",
                    )
                )
                plt.savefig(
                    os.path.join(
                        output_dir,
                        f"KRG_validation_{function.name}_{num_samples}_samples.eps",
                    )
                )
                plt.close()

            # Plot results if 2D
            if n_dim == 2 and PLOT_DIR:

                def _create_subplot(
                    ax: plt.Axes,
                    fig: plt.Figure,
                    x: list | np.ndarray,
                    y: list | np.ndarray,
                    title: str,
                ) -> None:
                    """Helper function to create a subplot."""
                    contour = ax.contourf(
                        x[:, 0].reshape(100, 10),
                        x[:, 1].reshape(100, 10),
                        y.reshape(100, 10),
                        levels=20,
                        cmap="viridis",
                    )
                    ax.set_title(title)
                    ax.set_xlabel("X1")
                    ax.set_ylabel("X2")
                    fig.colorbar(contour, ax=ax)
                    return

                fig, axs = plt.subplots(2, 2)
                ax1, ax2, ax3, ax4 = axs.flatten()

                # Plot the SMT KRG predictions top left
                _create_subplot(ax1, fig, x_eval, y_pred_smt, "SMT KRG Predictions")

                # Plot the Custom KRG predictions top right
                _create_subplot(
                    ax2, fig, x_eval, y_pred_custom, "Custom KRG Predictions"
                )

                # Plot the SMT KRG variance bottom left
                _create_subplot(ax3, fig, x_eval, y_var_smt, "SMT KRG Variance")

                # Plot the Custom KRG variance bottom right
                _create_subplot(ax4, fig, x_eval, y_var_custom, "Custom KRG Variance")

                plt.suptitle(
                    f"KRG Surrogate Model " f"{function.name} ({num_samples} samples)"
                )

                # Adjust margins
                fig.tight_layout()
                fig.subplots_adjust(top=0.88)

                # save the plot
                plt.savefig(
                    os.path.join(
                        output_dir,
                        f"KRG_validation_2D_{function.name}_{num_samples}_samples.png",
                    )
                )
                plt.savefig(
                    os.path.join(
                        output_dir,
                        f"KRG_validation_2D_{function.name}_{num_samples}_samples.eps",
                    )
                )
                plt.close()

            # Assert test
            np.testing.assert_allclose(y_pred_custom, y_pred_smt)
            np.testing.assert_allclose(y_var_custom, y_var_smt)

        Config.destroy_instance()
