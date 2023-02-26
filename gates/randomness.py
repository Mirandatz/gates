import numpy as np

RNG = np.random.Generator


def get_fixed_seed() -> int:
    return 42


def create_rng(seed: int | None = None) -> RNG:
    if seed is None:
        seed = get_fixed_seed()

    bit_gen = np.random.SFC64(seed=seed)
    return np.random.Generator(bit_gen)
