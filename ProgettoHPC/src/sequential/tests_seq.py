from .luby_seq import luby_sequential
from ..common.validate import is_independent_set, is_maximal
from ..common.graph_utils import (
    make_toy_graph,
    check_undirected,
    generate_erdos_renyi,
    generate_path,
    generate_star
)

def assert_mis_properties(adj, mis):
    assert is_independent_set(adj, mis), "FAIL: MIS is not independent"
    assert is_maximal(adj, mis), "FAIL: MIS is not maximal"

def run_tests():
    # 1) toy graph
    g = make_toy_graph()
    assert check_undirected(g), "FAIL: toy graph is not undirected"
    mis = luby_sequential(g, seed=0)
    assert_mis_properties(g, mis)

    # 2) path
    g = generate_path(20)
    assert check_undirected(g), "FAIL: path graph is not undirected"
    for s in range(5):
        mis = luby_sequential(g, seed=s)
        assert_mis_properties(g, mis)

    # 3) star
    g = generate_star(20)
    assert check_undirected(g), "FAIL: star graph is not undirected"
    for s in range(5):
        mis = luby_sequential(g, seed=s)
        assert_mis_properties(g, mis)

    # 4) random small ER (fast)
    g = generate_erdos_renyi(50, 0.1, seed=1)
    assert check_undirected(g), "FAIL: ER graph is not undirected"
    for s in range(5):
        mis, rounds = luby_sequential(g, seed=s, return_rounds=True)
        assert rounds > 0

    print("✅ All sequential tests PASSED")

if __name__ == "__main__":
    run_tests()