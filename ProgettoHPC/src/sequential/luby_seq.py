# Baseline sequenziale dell’algoritmo di Luby per calcolare un Maximal Independent Set (MIS)

import random
import time
from typing import Dict, Set, Tuple, Union, Any

# Tipo alias per rappresentare un grafo come lista di adiacenza:
Adj = Dict[int, Set[int]]


def luby_sequential(
    adj: Adj,
    seed: int = 0,
    return_rounds: bool = False,
    return_stats: bool = False,
    prefer_sorted_active: bool = True,
) -> Union[
    Set[int],
    Tuple[Set[int], int],
    Tuple[Set[int], int, Dict[str, Any]],
]:
    """
    Implementazione sequenziale dell'algoritmo di Luby per il MIS.

    Parametri:
    - adj: grafo non orientato (lista di adiacenza)
    - seed: seme RNG
    - return_rounds: se True, restituisce anche il numero di round
    - return_stats: se True, restituisce anche statistiche di timing interne
    - prefer_sorted_active: se True usa sorted(active) per riproducibilità più stabile

    Ritorna:
    - mis
    - (mis, rounds) se return_rounds=True
    - (mis, rounds, stats) se return_stats=True (rounds incluso)
    """
    rng = random.Random(seed)
    active: Set[int] = set(adj.keys())
    mis: Set[int] = set()
    rounds = 0

    # Statistiche tempi (somma su tutti i round)
    stats = {
        "t_total": 0.0,
        "t_prio": 0.0,
        "t_select": 0.0,
        "t_update": 0.0,
        "n_rounds": 0,
    }

    t_total0 = time.perf_counter()

    while active:
        rounds += 1

        # ----- 1) PRIO -----
        t0 = time.perf_counter()
        active_list = sorted(active) if prefer_sorted_active else list(active)
        prio = {v: (rng.random(), v) for v in active_list}
        t1 = time.perf_counter()
        stats["t_prio"] += (t1 - t0)

        # ----- 2) SELECT (minimi locali) -----
        t0 = time.perf_counter()
        selected: Set[int] = set()
        for v in active_list:
            neigh = adj[v].intersection(active)
            if all(prio[v] < prio[w] for w in neigh):
                selected.add(v)
        t1 = time.perf_counter()
        stats["t_select"] += (t1 - t0)

        # ----- 3) UPDATE (MIS + rimozioni) -----
        t0 = time.perf_counter()
        mis.update(selected)

        to_remove = set(selected)
        for v in selected:
            to_remove.update(adj[v].intersection(active))
        active.difference_update(to_remove)
        t1 = time.perf_counter()
        stats["t_update"] += (t1 - t0)

    stats["t_total"] = time.perf_counter() - t_total0
    stats["n_rounds"] = rounds

    if return_stats:
        return mis, rounds, stats
    if return_rounds:
        return mis, rounds
    return mis