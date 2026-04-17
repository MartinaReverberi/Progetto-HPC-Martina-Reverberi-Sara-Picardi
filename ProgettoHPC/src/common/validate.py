# verificare correttezza: indipendenza e massimalità

from typing import Dict, Set

# Tipo del grafo: dizionario nodo -> insieme vicini
Adj = Dict[int, Set[int]]

def is_independent_set(adj: Adj, nodes: Set[int]) -> bool:
    """
    Verifica che 'nodes' sia un insieme indipendente.

    Un insieme è indipendente se nessuna coppia di nodi
    nell'insieme è collegata da un arco.
    """
    for v in nodes:
        # Se uno dei vicini di v è anch'esso in nodes,
        # allora non è indipendente.
        if adj[v].intersection(nodes):
            return False
    return True

def is_maximal(adj: Adj, indep: Set[int]) -> bool:
    """
    Verifica che 'indep' sia massimale.

    Un insieme indipendente è massimale se
    ogni nodo fuori dall'insieme ha almeno
    un vicino nell'insieme.
    """
    for v in adj.keys():
        # Se il nodo è già nell'insieme, lo saltiamo.
        if v in indep:
            continue

        # Se v NON ha vicini nell'insieme,
        # allora potremmo aggiungerlo → non è massimale.
        if not adj[v].intersection(indep):
            return False
    return True