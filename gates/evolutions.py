import numpy as np
import numpy.typing as npt
from loguru import logger

import gates.fitnesses as gf
import gates.mutations as gm
import gates.nor_gates as ng
import gates.randomness as rand


def run_single_generation(
    population: list[gf.EvalutedIndividual],
    mutation_params: gm.MutationParameters,
    blacklist: set[ng.NorClassifier],
    features: npt.NDArray[np.bool_],
    targets: npt.NDArray[np.bool_],
    rng: rand.RNG,
) -> list[gf.EvalutedIndividual] | None:
    mutants = gm.try_generate_many_novel_mutants(
        population=[evaluated_individual[0] for evaluated_individual in population],
        params=mutation_params,
        blacklist=blacklist,
        rng=rng,
    )

    if mutants is None:
        return None

    evaluated_mutants = [
        (m, gf.evaluate_fitness(m, features, targets)) for m in mutants
    ]

    next_gen_candidates = population + evaluated_mutants

    fittest = gf.select_fittest(
        individuals=next_gen_candidates,
        fittest_count=len(population),
    )

    return fittest


def run_evolutionary_loop(
    population: list[gf.EvalutedIndividual],
    n_generations: int,
    mutations_params: gm.MutationParameters,
    features: npt.NDArray[np.bool_],
    targets: npt.NDArray[np.bool_],
    rng: rand.RNG,
) -> list[gf.EvalutedIndividual]:
    assert n_generations >= 1

    blacklist = set(evaluated_individual[0] for evaluated_individual in population)

    for gen_nr in range(n_generations):
        maybe_population = run_single_generation(
            population,
            mutation_params=mutations_params,
            blacklist=blacklist,
            features=features,
            targets=targets,
            rng=rng,
        )

        if maybe_population is None:
            break
        else:
            population = maybe_population

        best = max(population, key=lambda i_f: i_f[1])

        logger.info(f"generation {gen_nr}, best fitness={best[1]}")

    return population
