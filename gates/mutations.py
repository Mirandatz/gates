from numba.typed import List as NumbaList

import gates.nor_gates as ng
import gates.randomness as rand


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


def generate_mutants(
    population: list[ng.NorClassifier],
    n_mutants: int,
    rng: rand.RNG,
) -> list[ng.NorClassifier]:
    assert n_mutants >= 1

    all_mutants: list[ng.NorClassifier] = []

    while len(all_mutants) != n_mutants:
        mutation_candidate_index = rng.integers(low=0, high=len(population))
        mutation_candidate = population[mutation_candidate_index]
        generated_mutant = mutate_individual(mutation_candidate, rng)
        all_mutants.append(generated_mutant)

    return all_mutants
