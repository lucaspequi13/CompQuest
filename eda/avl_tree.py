"""Árvore AVL ordenada por timestamp, com soma de pontos por subárvore.

Histórico de cada condutor: placa -> AVL. Permite somar os pontos dos últimos
12 meses com uma consulta por intervalo em O(log n), sem varrer o histórico.
"""
from typing import Any, List, Optional, Tuple


class _Node:
    __slots__ = ("key", "value", "points", "left", "right", "height", "sum")

    def __init__(self, key: float, value: Any, points: int):
        self.key, self.value, self.points = key, value, points
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None
        self.height = 1
        self.sum = points  # soma de pontos da subárvore


def _h(n): return n.height if n else 0
def _s(n): return n.sum if n else 0


def _fix(n: _Node) -> None:
    n.height = 1 + max(_h(n.left), _h(n.right))
    n.sum = n.points + _s(n.left) + _s(n.right)


def _rot_right(y: _Node) -> _Node:
    x = y.left
    y.left, x.right = x.right, y
    _fix(y); _fix(x)
    return x


def _rot_left(x: _Node) -> _Node:
    y = x.right
    x.right, y.left = y.left, x
    _fix(x); _fix(y)
    return y


def _balance(n: _Node) -> _Node:
    _fix(n)
    bf = _h(n.left) - _h(n.right)
    if bf > 1:
        if _h(n.left.left) < _h(n.left.right):
            n.left = _rot_left(n.left)
        return _rot_right(n)
    if bf < -1:
        if _h(n.right.right) < _h(n.right.left):
            n.right = _rot_right(n.right)
        return _rot_left(n)
    return n


class AVLTree:
    def __init__(self):
        self.root: Optional[_Node] = None
        self.size = 0

    def insert(self, key: float, value: Any = None, points: int = 0) -> None:
        def ins(n):
            if n is None:
                return _Node(key, value, points)
            if key < n.key:
                n.left = ins(n.left)
            else:  # chaves iguais vão para a direita (permite duplicatas)
                n.right = ins(n.right)
            return _balance(n)
        self.root = ins(self.root)
        self.size += 1

    def height(self) -> int:
        return _h(self.root)

    def _sum_ge(self, n, lo) -> int:
        total = 0
        while n:
            if n.key >= lo:
                total += n.points + _s(n.right)
                n = n.left
            else:
                n = n.right
        return total

    def _sum_gt(self, n, hi) -> int:
        total = 0
        while n:
            if n.key > hi:
                total += n.points + _s(n.right)
                n = n.left
            else:
                n = n.right
        return total

    def sum_points(self, lo: float, hi: float) -> int:
        """Soma de pontos com lo <= key <= hi. O(log n)."""
        return self._sum_ge(self.root, lo) - self._sum_gt(self.root, hi)

    def range(self, lo: float, hi: float) -> List[Tuple[float, Any]]:
        out: List[Tuple[float, Any]] = []

        def walk(n):
            if not n:
                return
            if n.key >= lo:
                walk(n.left)
            if lo <= n.key <= hi:
                out.append((n.key, n.value))
            if n.key <= hi:
                walk(n.right)
        walk(self.root)
        return out
