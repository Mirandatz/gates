import numpy as np
import pytest
from numba.typed import List as NumbaList

import gates.nor_gates as ng


@pytest.mark.parametrize(
    "features, indices, expected",
    [
        ([True], [True], False),
        ([False], [True], True),
        # varying features, all-true indices
        ([False, False], [True, True], True),
        ([False, True], [True, True], False),
        ([True, False], [True, True], False),
        ([True, True], [True, True], False),
        # varying indices, all-true inputs
        ([True, True], [False, False], True),
        ([True, True], [False, True], False),
        ([True, True], [True, False], False),
        ([True, True], [True, True], False),
        # varying both, all-false inputs
        ([False, False], [False, False], True),
        ([False, False], [False, True], True),
        ([False, False], [True, False], True),
        ([False, False], [True, True], True),
        # bool overflow
        ([True] * 254, [True] * 254, False),
        ([True] * 255, [True] * 255, False),
        ([True] * 256, [True] * 256, False),
        ([True] * 257, [True] * 257, False),
        ([False] * 254, [True] * 254, True),
        ([False] * 255, [True] * 255, True),
        ([False] * 256, [True] * 256, True),
        ([False] * 257, [True] * 257, True),
    ],
)
def test_gate_predict_instance(
    features: list[bool],
    indices: list[bool],
    expected: bool,
) -> None:
    gate = ng.NorGate(np.asarray(indices))
    predicted = gate.predict_instance(np.asarray(features))
    assert expected == predicted


def test_nor_classifier_predict_instance() -> None:
    features = np.asarray([1, 1, 1], np.bool_)

    gates = [
        ng.NorGate(np.asarray([1, 1, 1], dtype=np.bool_)),
        ng.NorGate(np.asarray([1, 1, 1], dtype=np.bool_)),
        ng.NorGate(np.asarray([1, 1, 1, 1, 1], dtype=np.bool_)),
    ]

    classifier = ng.NorClassifier(
        gates=NumbaList(gates),
        num_features=3,
        num_classes=3,
    )

    predicted = classifier.predict_instance(features)
    expected = np.asarray([0, 0, 0], np.bool_)
    assert np.array_equal(expected, predicted)


def test_nor_gates_hash_and_eq() -> None:
    g1 = ng.NorGate(np.asarray([0, 0, 0], np.bool_))
    g2 = ng.NorGate(np.asarray([0, 0, 0], np.bool_))
    g3 = ng.NorGate(np.asarray([0, 1, 0], np.bool_))

    assert hash(g1) == hash(g2)
    assert g1 == g2
    assert hash(g1) != hash(g3)
    assert g1 != g3
