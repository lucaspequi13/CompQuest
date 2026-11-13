"""Fenwick tree (Binary Indexed Tree) para contagem de infrações por hora/região.

"Quantas infrações no cruzamento X entre 17h e 19h?" em O(log n).
Uma Fenwick por câmera; índice = minuto do dia (0..1439) ou hora (0..23).
"""
from typing import List


class Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.t: List[int] = [0] * (n + 1)

    def add(self, i: int, delta: int = 1) -> None:
        i += 1
        while i <= self.n:
            self.t[i] += delta
            i += i & -i

    def prefix(self, i: int) -> int:
        """Soma de [0, i]."""
        i += 1
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i
        return s

    def range_sum(self, lo: int, hi: int) -> int:
        return self.prefix(hi) - (self.prefix(lo - 1) if lo > 0 else 0)
