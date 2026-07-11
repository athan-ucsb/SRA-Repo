import random
import numpy as np

_rng = np.random.default_rng()

def seed_all(seed):
    _rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
    random.seed(seed)