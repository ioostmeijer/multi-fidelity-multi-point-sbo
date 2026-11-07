"""The file containing the constants used."""

# The functions delays used for the MF functions
MF2_FUNCTION_DELAY = {
    "forrester": [12 * 10, 1.2 * 10],
    "bohachevsky": [12 * 17, 1.2 * 17],
    "booth": [12 * 16, 1.2 * 16],
    "branin": [12 * 19, 1.2 * 19],
    "currin": [12 * 24, 1.2 * 24],
    "himmelblau": [12 * 21, 1.2 * 21],
    "six_hump_camelback": [12 * 37, 1.2 * 37],
    "park91a": [12 * 50, 1.2 * 50],
    "park91b": [12 * 126, 1.2 * 126],
    "hartmann6": [12 * 190, 1.2 * 190],
    "borehole": [12 * 550, 1.2 * 550],
}

# Combine all function delays
ALL_FUNCTION_DELAYS = {**MF2_FUNCTION_DELAY}
