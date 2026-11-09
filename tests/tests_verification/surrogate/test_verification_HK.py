import numpy as np
import pytest

from sbo import SBO, Config
from sbo.benchmarks import FORRESTER as function_class


@pytest.mark.verification
def test_LF_equal_KRG() -> None:
    """Test the LF HK prediction and var are equal to a KRG model."""
    x_test = np.linspace(function_class.l_bound, function_class.u_bound, 1000)

    config = Config(
        used_algorithms={
            "surrogate": "HK",
            "infill": "TwoStep",
            "optimisation": "EgoSequential",
        },
        lb=function_class.l_bound,
        ub=function_class.u_bound,
        functions=function_class.functions,
    )
    sbo = SBO(config)
    sbo.run()
    optimisation_data = sbo._optimisation._optimisation_data.copy()

    # calculate infill values for every iteration
    for key, key_value in optimisation_data.items():
        if isinstance(key, int) or key.isdigit():
            # train the surrogate on the samples
            samples_list = []
            for fid_samples in key_value["sample_points"].values():
                samples_list.extend(fid_samples)
            key_value["sample_points_list"] = samples_list

            sbo._surrogate.set_training_data(samples_list)
            sbo._surrogate.train()

            # Calculate predictions at lowest fidelity
            mu_lf = sbo._surrogate.predict_values(x_test, fidelity=1)
            var_lf = sbo._surrogate.predict_variance_fidelity(x_test, fidelity=1)
            key_value["mu_lf"] = mu_lf
            key_value["var_lf"] = var_lf

    # Destroy config instance
    Config.destroy_instance(destroy_logger=True)

    # Create instances with KRG surrogate and EI infill
    config = Config(
        used_algorithms={
            "surrogate": "KRG",
            "infill": "EI",
            "optimisation": "EgoSequential",
        },
        lb=function_class.l_bound,
        ub=function_class.u_bound,
        functions=[function_class.functions[-1]],
    )
    sbo = SBO(config)

    # Check against the HK values
    for key, key_value in optimisation_data.items():
        if isinstance(key, int) or key.isdigit():
            # train the surrogate
            # Only on LF samples, but they should be marked as HF
            training_data = [{**p, "fid": 0} for p in key_value["sample_points"][1]]
            sbo._surrogate.set_training_data(training_data)
            sbo._surrogate.train()

            # calculate the mu and var
            mu_krg = sbo._surrogate.predict_values(x_test)
            var_krg = sbo._surrogate.predict_variance(x_test)

            # Assert the lists are equal
            mu_diff = np.abs(np.array(mu_krg) - np.array(key_value["mu_lf"]))
            var_diff = np.abs(np.array(var_krg) - np.array(key_value["var_lf"]))

            max_mu_diff = max(mu_diff)
            max_var_diff = max(var_diff)
            max_rel_mu_diff = max(mu_diff / np.array(key_value["mu_lf"]))
            max_rel_var_diff = max(var_diff / np.array(key_value["var_lf"]))

            assert np.allclose(
                np.array(mu_krg), np.array(key_value["mu_lf"]), atol=1e-5, rtol=1e-3
            ), (
                f"For {key} of function {function_class.name} the mu values are"
                f" not equal.\nMax mu diff: {max_mu_diff} (abs), "
                f"{max_rel_mu_diff}(rel)\n"
            )
            assert np.allclose(
                np.array(var_krg), np.array(key_value["var_lf"]), atol=1e-5, rtol=1e-3
            ), (
                f"For {key} of function {function_class.name} the var values are"
                f"not equal.\n"
                f"Max var diff: {max_var_diff} (abs)\n"
                f"{max_rel_var_diff}(rel)\n"
            )

    # Destroy instance
    Config.destroy_instance(destroy_logger=True)
