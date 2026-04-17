"""
Test di correttezza per Luby parallelo (Joblib).

Nota: per evitare warning di multiprocessing su macOS/conda (resource_tracker),
qui usiamo backend="threading". Per i benchmark sul cluster userai "loky".
"""

from .luby_par import luby_joblib
from ..common.validate import is_independent_set, is_maximal
from ..common.graph_utils import (
    make_toy_graph,
    generate_path,
    generate_star,
    generate_erdos_renyi,
    check_undirected
)

# Backend usato SOLO nei test (più "pulito" su Mac)
TEST_BACKEND = "threading"


def assert_mis(adj, mis):
    """Verifica che mis sia indipendente e massimale."""
    assert is_independent_set(adj, mis), "FAIL: MIS is not independent"
    assert is_maximal(adj, mis), "FAIL: MIS is not maximal"


def run_tests():
    # 1) toy graph
    print("[TEST] toy graph")
    g = make_toy_graph()
    assert check_undirected(g), "FAIL: toy graph is not undirected"
    for s in range(5):
        mis = luby_joblib(g, seed=s, n_jobs=2, backend=TEST_BACKEND)
        assert_mis(g, mis)

    # 2) path
    print("[TEST] path graph")
    g = generate_path(50)
    assert check_undirected(g), "FAIL: path graph is not undirected"
    for s in range(5):
        mis = luby_joblib(g, seed=s, n_jobs=2, backend=TEST_BACKEND)
        assert_mis(g, mis)

    # 3) star
    print("[TEST] star graph")
    g = generate_star(50)
    assert check_undirected(g), "FAIL: star graph is not undirected"
    for s in range(5):
        mis = luby_joblib(g, seed=s, n_jobs=2, backend=TEST_BACKEND)
        assert_mis(g, mis)

    # 4) random ER
    print("[TEST] Erdős–Rényi small")
    g = generate_erdos_renyi(100, 0.05, seed=1)
    assert check_undirected(g), "FAIL: ER graph is not undirected"
    for s in range(5):
        mis, rds = luby_joblib(g, seed=s, n_jobs=2, backend=TEST_BACKEND, return_rounds=True)
        assert rds > 0, "FAIL: rounds should be > 0"
        assert_mis(g, mis)

    print("✅ All parallel tests PASSED")


if __name__ == "__main__":
    run_tests()