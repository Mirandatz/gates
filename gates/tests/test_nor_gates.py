import numpy as np
import pytest

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
        # bool8 overflow
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
    predicted = gate.predict(np.asarray(features))
    assert expected == predicted


@pytest.mark.xfail(reason="Not implemented")
def test_nor_classifier_predict_instance() -> None:
    features = np.asarray([1, 1, 1], np.bool8)

    gates = [
        ng.NorGate(np.asarray([1, 1, 1], dtype=np.bool8)),
        ng.NorGate(np.asarray([1, 1, 1], dtype=np.bool8)),
        ng.NorGate(np.asarray([1, 1, 1, 1, 1], dtype=np.bool8)),
    ]
