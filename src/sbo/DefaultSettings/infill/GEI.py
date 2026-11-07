"""Default settings used in the GEI and Filtered GEI infill strategy.

Keys:
        - g: int, factor determining relative importance between exploration and
        exploitation. Higher g -> more global exploration
        (default: 1, must be >= 0)
"""

GEI_DEFAULT_SETTINGS = {
    "g": {"type": int, "default": 1, "min": 0},
}
