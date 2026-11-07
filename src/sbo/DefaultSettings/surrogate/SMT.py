"""Default settings for the surrogate models from the SMT (surrogate Modelling Toolbox).

Keys (alphabetically ordered, spaces replaced with underscores):
    - 'print_global': bool, Global print toggle
        (default: False)
    - 'print_prediction': bool, whether to print prediction information
        (default: False)
    - 'print_problem': bool, whether to print problem information
        (default: False)
    - 'print_solver': bool, whether to print solver information
        (default: False)
    - 'print_training': bool, whether to print training information
        (default: False)
"""

SMT_SURROGATE_DEFAULT_SETTINGS = {
    "print_global": {"type": bool, "default": False},
    "print_prediction": {"type": bool, "default": False},
    "print_problem": {"type": bool, "default": False},
    "print_solver": {"type": bool, "default": False},
    "print_training": {"type": bool, "default": False},
}
