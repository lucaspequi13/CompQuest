"""Benchmarks para a apresentação: nossa estrutura vs. abordagem ingênua.

python benchmarks/bench_eda.py
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eda.avl_tree import AVLTree  # noqa: E402
from eda.priority_heap import MinHeap  # noqa: E402
from eda.quadtree import QuadTree, Rect  # noqa: E402


def timeit(fn, reps=1):
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1000


def bench_spatial(n=20000, queries=2000):
    rnd = random.Random(0)
    pts = [(rnd.uniform(0, 1920), rnd.uniform(0, 1080), i) for i in range(n)]
    qt = QuadTree(Rect(0, 0, 1920, 1080), capacity=16)
    for x, y, i in pts:
        qt.insert(x, y, i)
    areas = [Rect(rnd.uniform(0, 1800), rnd.uniform(0, 1000), 100, 80) for _ in range(queries)]
    lin = timeit(lambda: [[p for p in pts if a.contains(p[0], p[1])] for a in areas])
    qtt = timeit(lambda: [qt.query(a) for a in areas])
    print(f"Busca espacial ({n} pontos, {queries} consultas): lista {lin:.0f} ms | quadtree {qtt:.0f} ms | {lin / qtt:.0f}x")


def bench_points(n=50000, queries=2000):
    rnd = random.Random(1)
    evs = sorted((rnd.uniform(0, 1e8), rnd.choice([3, 4, 5, 7])) for _ in range(n))
    t = AVLTree()
    for ts, p in evs:
        t.insert(ts, None, p)
    qs = [(lo, lo + 3e7) for lo in (rnd.uniform(0, 7e7) for _ in range(queries))]
    lin = timeit(lambda: [sum(p for ts, p in evs if lo <= ts <= hi) for lo, hi in qs])
    avl = timeit(lambda: [t.sum_points(lo, hi) for lo, hi in qs])
    print(f"Soma de pontos em janela ({n} eventos): varredura {lin:.0f} ms | AVL {avl:.1f} ms | {lin / avl:.0f}x")


def bench_queue(n=20000):
    rnd = random.Random(2)
    xs = [rnd.randint(0, 3) for _ in range(n)]

    def naive():
        q = []
        for x in xs:
            q.append(x)
            q.sort()
        while q:
            q.pop(0)

    def heap():
        h = MinHeap()
        for x in xs:
            h.push(x, x)
        while len(h):
            h.pop()
    a, b = timeit(naive), timeit(heap)
    print(f"Fila de prioridade ({n} infrações): lista+sort {a:.0f} ms | heap {b:.0f} ms | {a / b:.0f}x")


if __name__ == "__main__":
    bench_spatial()
    bench_points()
    bench_queue()
