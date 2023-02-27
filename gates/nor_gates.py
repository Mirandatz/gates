import numba
import numpy as np
import numpy.typing as npt
from numba.experimental import jitclass

import gates.randomness as rand


@jitclass([("_mask", numba.boolean[:])])
class NOrGate:
    def __init__(self, mask: npt.NDArray[np.bool8]) -> None:
        assert len(mask) > 0
        self._mask = mask.copy()

    def predict(self, features: npt.NDArray[np.bool8]) -> bool:
        assert len(self._mask) <= len(features)

        result = False

        for i in range(len(self._mask)):
            if self._mask[i]:
                result = result or features[i]

        return not result


@numba.njit(cache=True)  # type: ignore
def create_gate(mask_size: int, rng: rand.RNG) -> NOrGate:
    assert mask_size > 0

    random_values = rng.random(size=mask_size)
    mask = (random_values > 0.5).flatten()
    return NOrGate(mask)
