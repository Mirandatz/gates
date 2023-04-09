import numpy as np
import numpy.typing as npt
from loguru import logger

import gates.fitnesses as gf
import gates.mutations as gm
import gates.randomness as rand


def run_single_generation(
    population: list[gf.EvalutedIndividual],
    n_mutants: int,
    features: npt.NDArray[np.bool_],
    targets: npt.NDArray[np.bool_],
    rng: rand.RNG,
) -> list[gf.EvalutedIndividual]:
    mutants = gm.generate_mutants(
        population=[evaluated_individual[0] for evaluated_individual in population],
        n_mutants=n_mutants,
        rng=rng,
    )

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
    n_mutants: int,
    features: npt.NDArray[np.bool_],
    targets: npt.NDArray[np.bool_],
    rng: rand.RNG,
) -> list[gf.EvalutedIndividual]:
    assert n_generations >= 1
    assert n_mutants >= 1

    for gen_nr in range(n_generations):
        population = run_single_generation(
            population,
            n_mutants,
            features,
            targets,
            rng,
        )

        best = max(population, key=lambda i_f: i_f[1])

        logger.info(f"generation {gen_nr}, best fitness={best[1]}")

    return population
