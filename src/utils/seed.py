"""
Random seed and reproducibility utilities.
"""

import os
import random
import numpy as np
import torch


def seed_all(seed: int = 42):
    """
    Seed all random number generators for reproducible runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
