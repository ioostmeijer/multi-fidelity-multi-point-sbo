from ..ALLOWED_ENTRIES import ALLOWED_TWOSTEP_SINGLE_FID_INFILL_STRATEGIES

"""Default Settings used in the Two-Step infill strategy.

Keys:
- single_fid_infill: str, the single-fidelity infill strategy used.
    This infill strategy is used to determine the sample location
    in the first step of the two-step approach. The name of the infill strategy should
    be included as a string, e.g., 'EI' for Expected Improvement.
    (default: 'EI', allowed: ALLOWED_TWOSTEP_SINGLE_FID_INFILL_STRATEGIES, i.e. the
    infill strategies except TwoStep, CEI and FilteredGEI)
- num_jsd_points: int, the number of points used to estimate the
    Jensen-Shannon distance (JSD) between the single-fidelity and
    multi-fidelity models. A higher number of points can lead to a more
    accurate estimation of the JSD but increases computational cost.
    (default: 1000, must be >= 1)
- jsd_threshold: float, the threshold value for the Jensen-Shannon distance (JSD).
    If the estimated JSD between the HF and lower fidelity models exceeds this
    threshold, the lower fidelity model is marked as not accurate enough. The default
    value is set to 0.7, based on Garbo et al. Multi-Fidelity Adaptive Sampling for
    Surrogate-Based Optimization and Uncertainty Quantification, 2024.
    (default: 0.7, min: 0.0, max: 1.0)
"""

TWOSTEP_DEFAULT_SETTINGS = {
    "single_fid_infill": {
        "type": str,
        "default": "EI",
        "allowed": ALLOWED_TWOSTEP_SINGLE_FID_INFILL_STRATEGIES,
    },
    "num_jsd_points": {
        "type": int,
        "default": 1000,
        "min": 1,
    },
    "jsd_threshold": {
        "type": float,
        "default": 0.7,  # Based on default in Garbo et al. (2024)
        "min": 0.0,
        "max": 1.0,
    },
}
