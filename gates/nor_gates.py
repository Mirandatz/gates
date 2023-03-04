import typing

import numpy as np
from numba import boolean, njit  # type: ignore
from numba.experimental import jitclass
from numba.typed import List as NumbaList

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
        assert len(mask) >= 1
        self.mask = mask.copy()

    def predict(self, features: BoolArray) -> bool:
        """
        Predicts the value of a single dataset row.
        """
        assert len(self.mask) <= len(features)

        relevant_features = features[self.mask]
        ored_features = relevant_features.sum() >= 1
        return not ored_features


@jitclass
class NorClassifier:
    """
    Instances of this class are mutable due to technical limitations;
    they should be treated as if they were immutable.
    """

    gates: typing.List[NorGate]
    class_count: int

    def __init__(
        self,
        gates: typing.List[NorGate],
        class_count: int,
    ) -> None:
        assert len(gates) >= 1
        assert len(gates) >= class_count

        # todo: explain why we perform this validation (flat vs layered storage)
        for i in range(1, len(gates)):
            prev = gates[i - 1]
            curr = gates[i]

            assert len(prev.mask) <= len(curr.mask)

        self.gates = NumbaList(gates)
        self.class_count = class_count

    def predict(self, features: BoolArray) -> BoolArray:
        """
        Predicts the value of a single dataset row.
        """

        assert features.ndim == 1

        feature_count = len(features)

        # will store feature values and the output of the gates
        augmented_features = np.zeros(
            shape=feature_count + len(self.gates),
            dtype=np.bool8,
        )

        # copy the original features
        augmented_features[:feature_count] = features

        # feed features plus previous gate outputs to current gate
        for i, g in enumerate(self.gates):
            visible_features = augmented_features[: feature_count + i]
            gate_prediction = g.predict(visible_features)
            augmented_features[feature_count + i] = gate_prediction

        # return the output of the last gates
        return augmented_features[-self.class_count :]


@njit(cache=True)  # type: ignore
def create_gate(mask_size: int, rng: rand.RNG) -> NorGate:
    assert mask_size > 0

    random_values = rng.random(size=mask_size)
    mask = (random_values > 0.5).flatten()
    return NorGate(mask)


def main() -> None:
    features = np.asarray([1, 1, 1], np.bool8)

    gates = [
        NorGate(np.asarray([0], np.bool8)),
        NorGate(np.asarray([0, 1], np.bool8)),
        NorGate(np.asarray([0, 0, 0], np.bool8)),
    ]

    for g in gates:
        print(g.predict(features))

    c0 = NorClassifier(NumbaList(gates), 3)

    print(c0.predict(features))


if __name__ == "__main__":
    main()
