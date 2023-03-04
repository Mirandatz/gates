from numba import boolean, njit
from numba.experimental import jitclass

import gates.randomness as rand
from gates.type_aliases import BoolArray


@jitclass
class NorGate:
    """
    Instances of this class are mutable due to technical limitations;
    they should be treated as if they were immutable.
    """

    mask: boolean[:]

    def __init__(self, mask: BoolArray) -> None:
        assert len(mask) > 0
        self.mask = mask.copy()

    def predict(self, features: BoolArray) -> bool:
        assert len(self.mask) <= len(features)

        relevant_features = features[self.mask]
        ored_features = relevant_features.sum() >= 1
        return not ored_features


@njit(cache=True)  # type: ignore
def create_gate(mask_size: int, rng: rand.RNG) -> NorGate:
    assert mask_size > 0

    random_values = rng.random(size=mask_size)
    mask = (random_values > 0.5).flatten()
    return NorGate(mask)
