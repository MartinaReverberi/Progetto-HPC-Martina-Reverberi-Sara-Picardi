import time
import numpy as np
from typing import Dict, Set, Tuple, Union, List, Iterable, Any

from joblib import Parallel, delayed

# Tipo del grafo: dizionario nodo -> insieme vicini
Adj = Dict[int, Set[int]]


def _chunks(lst: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    """Divide un array NumPy in chunk consecutivi di dimensione chunk_size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def _select_local_minima_chunk_np(
    chunk: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    active_mask: np.ndarray,
    prio_arr: np.ndarray
) -> List[int]:
    """
    Worker function HPC: legge dagli array NumPy (memmap condivisa di Joblib)
    senza overhead di serializzazione.
    """
    selected = []
    for v in chunk:
        # Trova i vicini usando i puntatori CSR
        start, end = indptr[v], indptr[v+1]
        neighs = indices[start:end]
        
        v_prio = prio_arr[v]
        is_min = True
        
        for w in neighs:
            if active_mask[w]: # Se il vicino è ancora attivo
                w_prio = prio_arr[w]
                # Confronta priorità (con fallback sull'ID del nodo come tie-breaker)
                if w_prio < v_prio or (w_prio == v_prio and w < v):
                    is_min = False
                    break
                    
        if is_min:
            selected.append(v)
            
    return selected


def _dict_to_csr(adj: Dict[int, Set[int]], n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Converte il dizionario di adiacenza in formato CSR (array piatti)."""
    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    indices_list = []

    for i in range(n_nodes):
        neighs = list(adj.get(i, set()))
        indices_list.extend(neighs)
        indptr[i+1] = len(indices_list)
        
    indices = np.array(indices_list, dtype=np.int32)
    return indptr, indices


def luby_joblib(
    adj: Adj,
    seed: int = 0,
    n_jobs: int = -1,
    backend: str = "loky",
    batch_factor: int = 4,
    return_rounds: bool = False,
    return_stats: bool = False,
    prefer_sorted_active: bool = True, # Parametro mantenuto per compatibilità
) -> Union[
    Set[int],
    Tuple[Set[int], int],
    Tuple[Set[int], int, Dict[str, Any]],
    Tuple[Set[int], Dict[str, Any]],
]:
    """
    Luby MIS - versione parallela con Joblib e NumPy Shared Memory.
    """
    n_nodes = len(adj)
    mis: Set[int] = set()
    rounds = 0

    if n_nodes == 0:
        if return_rounds and return_stats: return mis, rounds, {}
        if return_rounds: return mis, rounds
        if return_stats: return mis, {}
        return mis

    # Stats aggregati
    stats: Dict[str, Any] = {
        "t_prio": 0.0,
        "t_select": 0.0,
        "t_update": 0.0,
        "active_sum": 0,
        "active_max": 0,
        "n_chunks_last": None,
        "chunk_size_last": None,
    }

    # 1. PREPARAZIONE DATI (Una sola volta all'inizio)
    indptr, indices = _dict_to_csr(adj, n_nodes)
    
    # Array booleano per i nodi attivi (True = attivo, False = rimosso)
    active_mask = np.ones(n_nodes, dtype=bool) 
    prio_arr = np.zeros(n_nodes, dtype=np.float32)

    # Inizializziamo il generatore Random di NumPy (molto più veloce del modulo random standard)
    rng = np.random.default_rng(seed)

    n_active_now = n_nodes

    # 2. CICLO PRINCIPALE
    while n_active_now > 0:
        rounds += 1
        stats["active_sum"] += n_active_now
        stats["active_max"] = max(stats["active_max"], n_active_now)

        # A) Generazione Priorità
        t0 = time.perf_counter()
        # Ottiene gli ID (indici) dei nodi ancora attivi
        active_nodes = np.where(active_mask)[0] 
        # Genera un array di float random e lo assegna solo ai nodi attivi
        prio_arr[active_nodes] = rng.random(size=n_active_now)
        t1 = time.perf_counter()
        stats["t_prio"] += (t1 - t0)

        # B) Chunking
        est_workers = 8 if n_jobs == -1 else max(1, n_jobs)
        n_chunks = max(1, est_workers * max(1, batch_factor))
        chunk_size = max(1, (n_active_now + n_chunks - 1) // n_chunks)
        chunks = list(_chunks(active_nodes, chunk_size))

        stats["n_chunks_last"] = len(chunks)
        stats["chunk_size_last"] = chunk_size

        # C) Selezione Parallela
        t2 = time.perf_counter()
        selected_lists: List[List[int]] = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_select_local_minima_chunk_np)(chunk, indptr, indices, active_mask, prio_arr)
            for chunk in chunks
        )
        t3 = time.perf_counter()
        stats["t_select"] += (t3 - t2)

        # D) Merge + Update
        t4 = time.perf_counter()
        selected_flat = []
        for s_list in selected_lists:
            selected_flat.extend(s_list)

        mis.update(selected_flat)

        # Aggiornamento super-veloce della maschera: disattiviamo i nodi selezionati e i loro vicini
        for v in selected_flat:
            active_mask[v] = False
            start, end = indptr[v], indptr[v+1]
            active_mask[indices[start:end]] = False # Vettorizzazione NumPy
            
        # Ricalcola quanti nodi sono ancora attivi per il prossimo ciclo
        n_active_now = np.count_nonzero(active_mask)
        t5 = time.perf_counter()
        stats["t_update"] += (t5 - t4)

    # Post-processing stats
    if return_stats:
        stats["rounds"] = rounds
        stats["active_mean"] = stats["active_sum"] / rounds if rounds > 0 else 0.0
        total = stats["t_prio"] + stats["t_select"] + stats["t_update"]
        stats["t_total_internal"] = total
        stats["seq_frac_est"] = (stats["t_update"] / total) if total > 0 else 0.0

    # Ritorni
    if return_rounds and return_stats:
        return mis, rounds, stats
    if return_rounds:
        return mis, rounds
    if return_stats:
        return mis, stats
    return mis