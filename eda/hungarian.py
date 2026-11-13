"""Algoritmo Húngaro (Kuhn-Munkres) para atribuição de custo mínimo. O(n^3).

Usado pelo tracker: linhas = tracks existentes, colunas = detecções do frame
atual, custo = 1 - IoU. Implementado do zero (versão com potenciais / e-maxx),
aceita matrizes retangulares.
"""
from typing import List, Sequence, Tuple

INF = float("inf")


def hungarian(cost: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """Retorna pares (linha, coluna) que minimizam o custo total."""
    if not cost or not cost[0]:
        return []
    n_rows, n_cols = len(cost), len(cost[0])
    transposed = n_rows > n_cols
    if transposed:  # o algoritmo exige linhas <= colunas
        cost = [list(col) for col in zip(*cost)]
        n_rows, n_cols = n_cols, n_rows

    n, m = n_rows, n_cols
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)    # p[j] = linha atribuída à coluna j (1-indexado)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:  # refaz o caminho aumentante
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)
