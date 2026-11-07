"""Run an optimisation defined in a JSON config file.

The JSON file uses a benchmark function from mf2, an optional dependency:
pip install "multi-fidelity-multi-point-sbo[benchmarks]"
"""

import json
from pathlib import Path

from sbo import SBO, Config

# Path to config json
CONFIG_JSON = Path(__file__).parent / "05_run_from_config.json"

# Load the config json file
with open(CONFIG_JSON, "r") as f:
    config_dict = json.load(f)

# Create config and sbo instances from the config
config = Config.from_json(config_dict)
sbo = SBO(config)

# Run the optimisation
optimum = sbo.run()
