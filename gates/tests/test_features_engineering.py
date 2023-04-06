import numpy as np

import gates.features_engineering as fe


def test_features_encoder() -> None:
    features = np.arange(9).reshape(3, 3)
    encoder = fe.FeaturesEncoder().fit(features)
    encoded = encoder.transform(features)

    expected = np.asarray(
        [
            [True, False, False, True, False, False, True, False, False],
            [False, True, False, False, True, False, False, True, False],
            [False, False, True, False, False, True, False, False, True],
        ]
    )

    assert np.array_equal(expected, encoded)
