from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 20260518


def set_all(seed: int = DEFAULT_SEED) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
