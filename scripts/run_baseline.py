import os

# Determinism locks (must be set before Python does much work)
os.environ["PYTHONHASHSEED"] = "0"

import random
import numpy as np

random.seed(0)
np.random.seed(0)

from pathlib import Path

OUT_DIR = Path("reports") / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

results_path = OUT_DIR / "results.json"
plot_path = OUT_DIR / "equity_curve.png"