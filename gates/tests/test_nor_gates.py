import numpy as np
import pytest

import gates.nor_gates as ng


@pytest.mark.parametrize(
    "features, indices, expected",
    [
        ([True], [True], False),
        ([False], [True], True),
        # varying features
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
    ],
)
def test_parametrized(
    features: list[bool],
    indices: list[bool],
    expected: bool,
) -> None:
    gate = ng.NOrGate(np.asarray(indices))
    predicted = gate.predict(np.asarray(features))
    assert expected == predicted
