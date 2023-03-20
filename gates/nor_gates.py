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

    def predict_instance(self, features: BoolArray) -> bool:
        """
        Predicts the label of a single dataset row.
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
    num_features: int
    num_classes: int

    def __init__(
        self,
        gates: typing.List[NorGate],
        num_features: int,
        num_classes: int,
    ) -> None:
        assert num_features >= 1
        assert num_classes >= 2
        assert len(gates) >= num_classes

        # todo: explain why we perform this validation (flat vs layered storage)
        for i in range(1, len(gates)):
            prev = gates[i - 1]
            curr = gates[i]

            assert len(prev.mask) <= len(curr.mask)

        assert len(gates[0].mask) <= num_features

        self.gates = NumbaList(gates)
        self.num_features = num_features
        self.num_classes = num_classes

    def predict_instance(self, features: BoolArray) -> BoolArray:
        """
        Predicts the labels of a single dataset row.
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
            gate_prediction = g.predict_instance(visible_features)
            augmented_features[feature_count + i] = gate_prediction

        # return the output of the last gates
        return augmented_features[-self.num_classes :]

    def predict_dataset(self, dataset: BoolArray) -> BoolArray:
        """
        Predicts the labels of all dataset rows.
        """

        assert dataset.ndim == 2

        num_instances, num_features = dataset.shape

        predicted_labels = np.empty(
            shape=(num_instances, self.num_classes),
            dtype=np.bool8,
        )

        for index in range(num_instances):
            instance_features = dataset[index]
            instance_labels = self.predict_instance(instance_features)
            predicted_labels[index] = instance_labels

        return predicted_labels


@njit(cache=True)  # type: ignore
def create_gate(mask_size: int, rng: rand.RNG) -> NorGate:
    assert mask_size > 0

    random_values = rng.random(size=mask_size)
    mask = (random_values > 0.5).flatten()
    return NorGate(mask)


@njit(cache=True)  # type: ignore
def create_classifier(
    num_gates: int,
    num_features: int,
    num_classes: int,
    rng: rand.RNG,
) -> NorClassifier:
    assert num_features >= 1
    assert num_classes >= 2
    assert num_gates >= num_classes

    gates = NumbaList()

    for i in range(num_gates):
        g = create_gate(mask_size=num_features + i, rng=rng)
        gates.append(g)

    return NorClassifier(
        gates=gates,
        num_features=num_features,
        num_classes=num_classes,
    )


def main() -> None:
    features = np.asarray([1, 1, 1], np.bool8)

    gates = [
        NorGate(np.asarray([0], np.bool8)),
        NorGate(np.asarray([0, 1], np.bool8)),
        NorGate(np.asarray([0, 0, 0], np.bool8)),
    ]

    for g in gates:
        print(g.predict_instance(features))

    c0 = NorClassifier(
        gates=NumbaList(gates), num_features=len(features), num_classes=3
    )

    print(c0.predict_instance(features))


if __name__ == "__main__":
    main()
