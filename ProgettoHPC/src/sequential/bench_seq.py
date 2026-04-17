import time
import statistics
import csv
from .luby_seq import luby_sequential
from ..common.graph_utils import make_toy_graph, generate_erdos_renyi, check_undirected
from ..common.validate import is_independent_set, is_maximal


def benchmark_seq(n_values, p_values, repeats=5, seed=0):
    results = []

    for n in n_values:
        for p in p_values:
            g = generate_erdos_renyi(n, p, seed=seed)
            assert check_undirected(g), "Graph is not undirected!"

            # warm-up
            luby_sequential(g, seed=seed)

            times = []
            sizes = []
            rounds_list = []

            t_prio_list = []
            t_select_list = []
            t_update_list = []

            ok_ind = True
            ok_max = True

            for r in range(repeats):
                s = seed + r

                t0 = time.perf_counter()
                mis, rds, stats = luby_sequential(
                    g, seed=s, return_stats=True
                )
                t1 = time.perf_counter()

                times.append(t1 - t0)
                sizes.append(len(mis))
                rounds_list.append(rds)

                t_prio_list.append(stats["t_prio"])
                t_select_list.append(stats["t_select"])
                t_update_list.append(stats["t_update"])

                ok_ind = ok_ind and is_independent_set(g, mis)
                ok_max = ok_max and is_maximal(g, mis)

            results.append({
                "n": n,
                "p": p,
                "m_est": int(p * n * (n - 1) / 2),
                "n_jobs": 1,
                "backend": "sequential",
                "batch_factor": "",
                "speedup": 1.0,
                "efficiency": 1.0,
                "time_mean": statistics.mean(times),
                "time_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
                "mis_mean": statistics.mean(sizes),
                "mis_min": min(sizes),
                "mis_max": max(sizes),
                "rounds_mean": statistics.mean(rounds_list),
                "rounds_min": min(rounds_list),
                "rounds_max": max(rounds_list),
                "t_prio_mean": statistics.mean(t_prio_list),
                "t_select_mean": statistics.mean(t_select_list),
                "t_update_mean": statistics.mean(t_update_list),
                "t_prio_std": statistics.pstdev(t_prio_list) if len(t_prio_list) > 1 else 0.0,
                "t_select_std": statistics.pstdev(t_select_list) if len(t_select_list) > 1 else 0.0,
                "t_update_std": statistics.pstdev(t_update_list) if len(t_update_list) > 1 else 0.0,
                "ok": ok_ind and ok_max,
            })

    return results


def print_table(results):
    print("n\tp\tmean(s)\tstd\tMIS_mean\trounds_mean\tOK")
    for r in results:
        print(
            f'{r["n"]}\t{r["p"]:.5f}\t'
            f'{r["time_mean"]:.6f}\t{r["time_std"]:.6f}\t'
            f'{r["mis_mean"]:.1f}\t{r["rounds_mean"]:.1f}\t'
            f'{r["ok"]}'
        )


def save_csv(results, filename="seq_results.csv"):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV to: {filename}")


def benchmark_rounds_vs_n(n_values, p=0.05, repeats=5, seed=0):
    results = []
    for n in n_values:
        g = generate_erdos_renyi(n, p, seed=seed)
        assert check_undirected(g), "Graph is not undirected!"
        luby_sequential(g, seed=seed)

        rounds_list = []
        for r in range(repeats):
            s = seed + r
            _, rds = luby_sequential(g, seed=s, return_rounds=True)
            rounds_list.append(rds)

        results.append({
            "n": n,
            "p": p,
            "rounds_mean": statistics.mean(rounds_list),
            "rounds_min": min(rounds_list),
            "rounds_max": max(rounds_list),
        })
    return results


def print_rounds_table(results):
    print("n\tp\trounds_mean\trounds[min,max]")
    for r in results:
        print(f'{r["n"]}\t{r["p"]:.3f}\t{r["rounds_mean"]:.1f}\t[{r["rounds_min"]},{r["rounds_max"]}]')


if __name__ == "__main__":
    # toy graph
    toy = make_toy_graph()
    mis, rds, stats = luby_sequential(toy, seed=42, return_stats=True)
    print("Toy MIS:", mis)
    print("Independent:", is_independent_set(toy, mis))
    print("Maximal:", is_maximal(toy, mis))
    print("Rounds:", rds)
    print("Stats:", stats)
    print()

    # -------------------------------------------------------------------------
    # Benchmark principale.
    # p bassissimo per Extra-Large per evitare OOM (Out Of Memory)
    # -------------------------------------------------------------------------
    configs = [
        ([200, 500, 1000, 5000, 10000], [0.01, 0.05, 0.1]),
        ([50000, 100000], [0.01, 0.05]),
        ([250000, 500000], [0.0001, 0.0005]) 
    ]

    print("=== Benchmark sequenziale ===")
    all_res = []
    
    for n_vals, p_vals in configs:
        print(f"\n--- Esecuzione batch sequenziale: n={n_vals}, p={p_vals} ---")
        repeats = 5 if max(n_vals) <= 10000 else 3
        res = benchmark_seq(n_vals, p_vals, repeats=repeats, seed=123)
        all_res.extend(res)
        # Salvataggio progressivo!
        save_csv(all_res, "seq_results.csv")

    print("\n=== RISULTATI FINALI SEQUENZIALE ===")
    print_table(all_res)

    print("\n--- Rounds vs n (fixed p=0.05) ---")
    rounds_res = benchmark_rounds_vs_n(
        [200, 500, 1000, 5000, 10000, 50000, 100000],
        p=0.05, repeats=3, seed=123
    )
    print_rounds_table(rounds_res)
    save_csv(rounds_res, "seq_rounds_vs_n.csv")