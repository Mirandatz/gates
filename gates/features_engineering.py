import numpy as np  # noqa
import numpy.typing as npt
from sklearn.datasets import load_iris  # noqa
from sklearn.preprocessing import KBinsDiscretizer  # noqa


def compute_thresholds(feature: npt.ArrayLike) -> npt.ArrayLike:
    feature = np.asarray(feature)
    assert feature.ndim == 1
    return np.unique(feature)  # type: ignore


def encode_single_feature(
    feature: npt.ArrayLike,
    thresholds: npt.ArrayLike,
) -> npt.NDArray[np.bool_]:
    feature = np.asarray(feature)
    thresholds = np.asarray(thresholds)

    assert feature.ndim == 1
    assert thresholds.ndim == 1

    values = np.digitize(feature, thresholds, right=True)
    identity = np.eye(N=thresholds.size, dtype=np.bool_)
    one_hot_encoded = identity[values]
    return one_hot_encoded  # type: ignore


class FeaturesEncoder:
    """
    This class performs a "one-hot-encoding" of continous (and discrete) features
    by creating `n` buckets, with n = the number of unique values for a feature,
    and bucketizing the values.
    """

    def __init__(self) -> None:
        self._all_thresholds: list[npt.ArrayLike] | None = None

    def fit(self, X: npt.ArrayLike, y: None = None) -> "FeaturesEncoder":
        features = np.asarray(X)

        assert features.ndim == 2

        n_samples, n_features = features.shape

        all_thresholds = []

        for feature_index in range(n_features):
            feature = features[:, feature_index]
            threshold = compute_thresholds(feature)
            all_thresholds.append(threshold)

        self._all_thresholds = all_thresholds

        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        assert self._all_thresholds is not None

        features = np.asarray(X)
        n_samples, n_features = features.shape

        assert len(self._all_thresholds) == n_features

        transformed = []

        for feature_index, feature_thresholds in enumerate(self._all_thresholds):
            feature_values = features[:, feature_index]
            encoded_feature = encode_single_feature(feature_values, feature_thresholds)
            transformed.append(encoded_feature)

        return np.hstack(transformed)

    def fit_transform(self, X: npt.ArrayLike, y: None = None) -> npt.NDArray[np.bool_]:
        self.fit(X, y)
        return self.transform(X)
