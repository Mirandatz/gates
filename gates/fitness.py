import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score

import gates.nor_gates as ng

EvalutedIndividual = tuple[ng.NorClassifier, float]


def evaluate_fitness(
    individual: ng.NorClassifier,
    features: npt.NDArray[np.bool8],
    targets: npt.NDArray[np.bool8],
) -> float:
    predicted = ng.parallel_predict_dataset(individual, features)
    accuracy = accuracy_score(targets, predicted)
    assert isinstance(accuracy, float)
    return accuracy


def select_fittest(
    individuals: list[EvalutedIndividual],
    fittest_count: int,
) -> list[EvalutedIndividual]:
    assert fittest_count <= len(individuals)

    from_best_to_worst = sorted(
        individuals,
        key=lambda ind_fit: ind_fit[1],
        reverse=True,
    )

    return from_best_to_worst[:fittest_count]
