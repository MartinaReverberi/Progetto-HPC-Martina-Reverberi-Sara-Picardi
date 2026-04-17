import math
import random
from typing import Dict, Set

# Tipo del grafo: dizionario nodo -> insieme vicini
Adj = Dict[int, Set[int]]

def make_toy_graph() -> Adj:
    """
    Grafo piccolo per test:
    - nodi 0,1,2 formano un triangolo (clique di 3)
    - nodo 3 è isolato
    """
    return {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1},
        3: set()
    }

def check_undirected(adj: Adj) -> bool:
    """
    Controlla che il grafo sia non orientato:
    se v è vicino di w allora w deve essere vicino di v.
    """
    for v, neigh in adj.items():
        for w in neigh:
            if v not in adj.get(w, set()): 
                return False
    return True

def generate_erdos_renyi(n: int, p: float, seed: int = 0) -> Adj:
    """
    Genera un grafo Erdős–Rényi G(n,p) usando l'algoritmo O(n + m) 
    di Batagelj e Brandes (2005). 
    Evita il doppio ciclo O(n^2) che causa timeout per grafi enormi.
    """
    rng = random.Random(seed)
    adj: Adj = {i: set() for i in range(n)}
    
    if p <= 0: return adj
    if p >= 1:
        for i in range(n):
            adj[i] = set(range(n)) - {i}
        return adj
        
    v = 1
    w = -1
    lp = math.log(1.0 - p)
    
    while v < n:
        lr = math.log(1.0 - rng.random())
        w = w + 1 + int(lr / lp)
        while w >= v and v < n:
            w = w - v
            v = v + 1
        if v < n:
            adj[v].add(w)
            adj[w].add(v)
            
    return adj

def generate_path(n: int) -> Adj:
    """
    Genera un path (catena):
    0-1-2-...-(n-1)
    """
    adj: Adj = {i: set() for i in range(n)}
    for i in range(n - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    return adj

def generate_star(n: int) -> Adj:
    """
    Genera una stella:
    - nodo 0 è il centro
    - tutti gli altri nodi 1..n-1 sono collegati al centro
    """
    adj: Adj = {i: set() for i in range(n)}
    for i in range(1, n):
        adj[0].add(i)
        adj[i].add(0)
    return adj