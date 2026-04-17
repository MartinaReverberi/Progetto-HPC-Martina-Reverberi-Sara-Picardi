import time
import statistics
import csv
from .luby_par import luby_joblib
from ..common.graph_utils import generate_erdos_renyi, check_undirected, make_toy_graph
from ..common.validate import is_independent_set, is_maximal

def _load_seq_times(seq_csv="seq_results.csv"):
    import os
    if not os.path.exists(seq_csv):
        return None
    seq_times = {}
    with open(seq_csv, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            key = (int(row["n"]), float(row["p"]))
            seq_times[key] = float(row["time_mean"])
    return seq_times

def _run_seq_fallback(g, n, p, repeats, seed):
    from ..sequential.luby_seq import luby_sequential
    import time as _time
    luby_sequential(g, seed=seed)  # warm-up
    times = []
    for r in range(repeats):
        t0 = _time.perf_counter()
        luby_sequential(g, seed=seed + r)
        t1 = _time.perf_counter()
        times.append(t1 - t0)
    return statistics.mean(times)

def benchmark_par(n_values, p_values, n_jobs_list, repeats=5, seed=0,
                  backend="loky", batch_factor=4, seq_csv="seq_results.csv"):
    results = []
    seq_times = _load_seq_times(seq_csv)
    
    if seq_times is None:
        print(f"WARNING: {seq_csv} non trovato. "
              "Calcolo T_seq con fallback (luby_sequential inline).\n"
              "Per risultati più precisi, esegui prima bench_seq.py.")

    for n in n_values:
        for p in p_values:
            g = generate_erdos_renyi(n, p, seed=seed)
            assert check_undirected(g), "Graph is not undirected!"

            key = (n, p)
            if seq_times is not None and key in seq_times:
                t_seq = seq_times[key]
            else:
                print(f"  T_seq non trovato per n={n}, p={p} → calcolo inline...")
                t_seq = _run_seq_fallback(g, n, p, repeats, seed)

            for n_jobs in n_jobs_list:
                # warm-up 
                luby_joblib(
                    g, seed=seed,
                    n_jobs=n_jobs, backend=backend,
                    batch_factor=batch_factor,
                    return_rounds=False,
                    return_stats=False
                )

                times = []
                sizes = []
                rounds_list = []
                t_prio_list = []
                t_select_list = []
                t_update_list = []
                active_mean_list = []
                active_max_list = []
                seq_frac_list = []
                ok_ind = True
                ok_max = True

                for r in range(repeats):
                    s = seed + r
                    t0 = time.perf_counter()
                    mis, rds, st = luby_joblib(
                        g, seed=s,
                        n_jobs=n_jobs, backend=backend,
                        batch_factor=batch_factor,
                        return_rounds=True,
                        return_stats=True
                    )
                    t1 = time.perf_counter()

                    times.append(t1 - t0)
                    sizes.append(len(mis))
                    rounds_list.append(rds)
                    t_prio_list.append(st["t_prio"])
                    t_select_list.append(st["t_select"])
                    t_update_list.append(st["t_update"])
                    active_mean_list.append(st["active_mean"])
                    active_max_list.append(st["active_max"])
                    seq_frac_list.append(st["seq_frac_est"])

                    ok_ind = ok_ind and is_independent_set(g, mis)
                    ok_max = ok_max and is_maximal(g, mis)

                t_par = statistics.mean(times)
                speedup = (t_seq / t_par) if t_par > 0 else 0.0
                efficiency = (speedup / n_jobs) if n_jobs > 0 else 0.0

                row = {
                    "n": n, "p": p, "n_jobs": n_jobs, "backend": backend,
                    "batch_factor": batch_factor, "m_est": int(p * n * (n - 1) / 2),
                    "time_mean": t_par,
                    "time_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
                    "time_seq_ref": t_seq,
                    "mis_mean": statistics.mean(sizes),
                    "mis_min": min(sizes), "mis_max": max(sizes),
                    "rounds_mean": statistics.mean(rounds_list),
                    "rounds_min": min(rounds_list), "rounds_max": max(rounds_list),
                    "t_prio_mean": statistics.mean(t_prio_list),
                    "t_select_mean": statistics.mean(t_select_list),
                    "t_update_mean": statistics.mean(t_update_list),
                    "active_mean": statistics.mean(active_mean_list),
                    "active_max": max(active_max_list),
                    "seq_frac_est_mean": statistics.mean(seq_frac_list),
                    "speedup": speedup, "efficiency": efficiency,
                    "ok": ok_ind and ok_max,
                }
                results.append(row)
    return results

def print_table(results):
    print("n\tp\tn_jobs\tmean(s)\tT_seq_ref(s)\tspeedup\teff\tMIS_mean\trounds_mean\tOK")
    for r in results:
        print(
            f'{r["n"]}\t{r["p"]:.5f}\t{r["n_jobs"]}\t'
            f'{r["time_mean"]:.6f}\t{r["time_seq_ref"]:.6f}\t'
            f'{r["speedup"]:.3f}\t{r["efficiency"]:.3f}\t'
            f'{r["mis_mean"]:.1f}\t{r["rounds_mean"]:.1f}\t{r["ok"]}'
        )

def save_csv(results, filename="par_results.csv"):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV to: {filename}")

if __name__ == "__main__":
    toy = make_toy_graph()
    mis, rds, st = luby_joblib(toy, seed=42, n_jobs=2, backend="loky", return_rounds=True, return_stats=True)
    print("Toy MIS (par):", mis)
    print("Independent:", is_independent_set(toy, mis))
    print("Maximal:", is_maximal(toy, mis))
    print()

    configs = [
        ([200, 500, 1000, 5000, 10000], [0.01, 0.05, 0.1]),
        ([50000, 100000], [0.01, 0.05]),
        ([250000, 500000], [0.0001, 0.0005]) 
    ]
    
    n_jobs_list = [1, 2, 4, 8]
    all_res = []

    print("=== Benchmark parallelo HPC (Backend: Loky) ===")
    for n_vals, p_vals in configs:
        print(f"\n--- Esecuzione batch: n={n_vals}, p={p_vals} ---")
        repeats = 5 if max(n_vals) <= 10000 else 3
        res = benchmark_par(
            n_values=n_vals, p_values=p_vals, n_jobs_list=n_jobs_list,
            repeats=repeats, seed=123, backend="loky", batch_factor=4,
            seq_csv="seq_results.csv"
        )
        all_res.extend(res)
        # Salvataggio progressivo!
        save_csv(all_res, "par_results.csv")

    print("\n=== RISULTATI FINALI ===")
    print_table(all_res)