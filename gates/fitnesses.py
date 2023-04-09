import typing

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score

import gates.nor_gates as ng

EvalutedIndividual = tuple[ng.NorClassifier, float]
Fitness = float


def evaluate_fitness(
    individual: ng.NorClassifier,
    features: npt.NDArray[np.bool_],
    targets: npt.NDArray[np.bool_],
) -> Fitness:
    predicted = ng.parallel_predict_dataset(individual, features)
    accuracy = accuracy_score(targets, predicted)
    assert isinstance(accuracy, float)
    return accuracy


T = typing.TypeVar("T")


def select_fittest(
    individuals: list[tuple[T, Fitness]],
    fittest_count: int,
) -> list[tuple[T, Fitness]]:
    assert fittest_count >= 1
    assert fittest_count <= len(individuals)

    from_best_to_worst = sorted(
        individuals,
        key=lambda ind_fit: ind_fit[1],
        reverse=True,
    )

    return from_best_to_worst[:fittest_count]
