import functools

import attrs
from numba.typed import List as NumbaList

import gates.fallible as fallible
import gates.nor_gates as ng
import gates.randomness as rand


@attrs.frozen
class MutationParameters:
    mutants_to_generate: int
    max_failures: int

    def __attrs_post_init__(self) -> None:
        assert self.mutants_to_generate >= 1
        assert self.max_failures >= 0


def modify_random_gene(individual: ng.NorClassifier, rng: rand.RNG) -> ng.NorClassifier:
    genes = individual.gates

    index_gene_to_mutate = rng.integers(low=0, high=len(genes))

    old_gene = genes[index_gene_to_mutate]
    new_gene = ng.create_gate(len(old_gene.mask), rng)

    new_genotype = NumbaList(genes)
    new_genotype[index_gene_to_mutate] = new_gene

    return ng.NorClassifier(
        new_genotype, individual.num_features, individual.num_classes
    )


def increase_genotype_size(
    individual: ng.NorClassifier,
    rng: rand.RNG,
) -> ng.NorClassifier:
    genes = individual.gates

    last_gate = genes[-1]

    new_gene = ng.create_gate(mask_size=len(last_gate.mask) + 1, rng=rng)
    new_genotype = NumbaList(genes)
    new_genotype.append(new_gene)

    return ng.NorClassifier(
        new_genotype, individual.num_features, individual.num_classes
    )


def decrease_genotype_size(
    individual: ng.NorClassifier,
    rng: rand.RNG,
) -> ng.NorClassifier:
    genes = individual.gates

    new_genotype = NumbaList(genes)
    new_genotype.pop()

    return ng.NorClassifier(
        new_genotype, individual.num_features, individual.num_classes
    )


def mutate_individual(individual: ng.NorClassifier, rng: rand.RNG) -> ng.NorClassifier:
    mutation_types = [
        modify_random_gene,
        increase_genotype_size,
        decrease_genotype_size,
    ]
    mutation_index = rng.integers(low=0, high=len(mutation_types))
    chosen_mutation = mutation_types[mutation_index]
    return chosen_mutation(individual, rng)


def try_generate_novel_mutant(
    population: list[ng.NorClassifier],
    blacklist: set[ng.NorClassifier],
    rng: rand.RNG,
) -> ng.NorClassifier | None:
    # not using rng.choice because type hints...
    mutation_candidate_index = rng.integers(low=0, high=len(population))
    mutation_candidate = population[mutation_candidate_index]

    mutant = mutate_individual(mutation_candidate, rng)

    if mutant in blacklist:
        return None

    blacklist.add(mutant)
    return mutant


def try_generate_many_novel_mutants(
    population: list[ng.NorClassifier],
    params: MutationParameters,
    blacklist: set[ng.NorClassifier],
    rng: rand.RNG,
) -> list[ng.NorClassifier] | None:
    assert len(population) >= 1

    # we only update the set of known individuals if we succeed
    blacklist_copy = set(blacklist)

    generator = functools.partial(
        try_generate_novel_mutant,
        population=population,
        blacklist=blacklist_copy,
        rng=rng,
    )

    results = fallible.collect_results_from_fallible_function(
        generator,
        num_results=params.mutants_to_generate,
        max_failures=params.max_failures,
    )

    if results is None:
        return None

    blacklist.update(blacklist_copy)
    return results
