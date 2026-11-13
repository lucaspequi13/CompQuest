"""Min-heap binário implementado do zero (sem heapq).

Fila de envio ao Detran: menor prioridade sai primeiro. Usamos
prioridade = -peso_da_gravidade, então gravíssima (peso 4) sai antes de leve (1).
Empate resolvido pela ordem de chegada (FIFO). push/pop O(log n).
"""
from typing import Any, List, Tuple


class MinHeap:
    def __init__(self):
        self._a: List[Tuple[float, int, Any]] = []
        self._seq = 0

    def __len__(self) -> int:
        return len(self._a)

    def push(self, priority: float, item: Any) -> None:
        self._a.append((priority, self._seq, item))
        self._seq += 1
        self._sift_up(len(self._a) - 1)

    def peek(self) -> Any:
        if not self._a:
            raise IndexError("heap vazio")
        return self._a[0][2]

    def pop(self) -> Any:
        if not self._a:
            raise IndexError("heap vazio")
        top = self._a[0]
        last = self._a.pop()
        if self._a:
            self._a[0] = last
            self._sift_down(0)
        return top[2]

    def _sift_up(self, i: int) -> None:
        a = self._a
        while i > 0:
            parent = (i - 1) // 2
            if a[i] < a[parent]:
                a[i], a[parent] = a[parent], a[i]
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        a, n = self._a, len(self._a)
        while True:
            l, r, smallest = 2 * i + 1, 2 * i + 2, i
            if l < n and a[l] < a[smallest]:
                smallest = l
            if r < n and a[r] < a[smallest]:
                smallest = r
            if smallest == i:
                break
            a[i], a[smallest] = a[smallest], a[i]
            i = smallest


SEVERITY_WEIGHT = {"leve": 1, "media": 2, "grave": 3, "gravissima": 4}


class ViolationQueue(MinHeap):
    def push_violation(self, violation: Any, severity: str) -> None:
        self.push(-SEVERITY_WEIGHT[severity], violation)
