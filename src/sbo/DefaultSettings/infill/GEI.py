"""Default settings used in the GEI and Filtered GEI infill strategy.

Keys:
    - g: int, the exponent $g$ of the GEI, which determines the relative importance
        between exploration and exploitation. A higher $g$ gives more global
        exploration; $g = 0$ is the probability of improvement and $g = 1$ the
        Expected Improvement
        (default: 1, must be >= 0)
"""

GEI_DEFAULT_SETTINGS = {
    "g": {"type": int, "default": 1, "min": 0},
}
