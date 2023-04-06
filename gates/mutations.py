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

    new_gene = ng.create_gate(mask_size=len(genes), rng=rng)
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


def mutate_genotype_size(
    individual: ng.NorClassifier,
    rng: rand.RNG,
) -> ng.NorClassifier:
    genes = individual.gates

    possible_mutation_types = ["increase_size"]
    if len(genes) > individual.num_classes:
        possible_mutation_types.append("decrease_size")

    chosen_mutation_type = rng.choice(possible_mutation_types)

    if chosen_mutation_type == "increase_size":
        return increase_genotype_size(individual, rng)
    else:
        return decrease_genotype_size(individual, rng)
