import gates.fitnesses as fits


def test_select_fittest() -> None:
    data = [
        ("a", 0),
        ("b", 1),
        ("c", 2),
        ("d", 1.5),
        ("a", 4),
    ]

    sorted_by_fitness = fits.select_fittest(data, fittest_count=2)
    expected = [("a", 4), ("c", 2)]

    assert expected == sorted_by_fitness
