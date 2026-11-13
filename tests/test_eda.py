import itertools
import random

from eda.avl_tree import AVLTree
from eda.bloom_filter import BloomFilter
from eda.geometry import iou, point_in_polygon, point_segment_distance, segments_intersect
from eda.hungarian import hungarian
from eda.plate_trie import PlateTrie, is_valid_plate, normalize
from eda.priority_heap import MinHeap, ViolationQueue
from eda.quadtree import QuadTree, Rect
from eda.ring_buffer import RingBuffer
from eda.road_graph import RoadGraph
from eda.segment_tree import Fenwick


def test_ring_buffer_keeps_last_n():
    rb = RingBuffer(3)
    for i in range(5):
        rb.push(i)
    assert rb.snapshot() == [2, 3, 4] and rb.latest() == 4


def test_geometry():
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
    assert not segments_intersect((0, 0), (1, 1), (2, 2), (3, 5))
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq) and not point_in_polygon((15, 5), sq)
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def _brute(cost):
    n, m = len(cost), len(cost[0])
    best = float("inf")
    if n <= m:
        for perm in itertools.permutations(range(m), n):
            best = min(best, sum(cost[i][perm[i]] for i in range(n)))
    else:
        for perm in itertools.permutations(range(n), m):
            best = min(best, sum(cost[perm[j]][j] for j in range(m)))
    return best


def test_hungarian_matches_brute_force():
    rnd = random.Random(0)
    for _ in range(40):
        n, m = rnd.randint(1, 5), rnd.randint(1, 5)
        cost = [[rnd.random() for _ in range(m)] for _ in range(n)]
        pairs = hungarian(cost)
        assert len(pairs) == min(n, m)
        assert abs(sum(cost[r][c] for r, c in pairs) - _brute(cost)) < 1e-9


def test_quadtree_equals_linear_scan():
    rnd = random.Random(1)
    qt = QuadTree(Rect(0, 0, 1920, 1080), capacity=4)
    pts = [(rnd.uniform(0, 1920), rnd.uniform(0, 1080), i) for i in range(500)]
    for x, y, i in pts:
        qt.insert(x, y, i)
    area = Rect(300, 200, 400, 300)
    got = sorted(d for _, _, d in qt.query(area))
    want = sorted(i for x, y, i in pts if area.contains(x, y))
    assert got == want


def test_road_graph():
    g = RoadGraph()
    g.allow("norte_entrada", "cruzamento")
    g.allow("cruzamento", "leste_saida")
    assert g.first_illegal_move(["norte_entrada", "cruzamento", "leste_saida"]) is None
    assert g.first_illegal_move(["leste_saida", "cruzamento"]) == ("leste_saida", "cruzamento")
    g.allow("A", "B", 5); g.allow("B", "C", 1); g.allow("A", "C", 10)
    assert g.shortest_path("A", "C") == (6.0, ["A", "B", "C"])


def test_heap_sorts_and_prioritizes():
    rnd = random.Random(2)
    h = MinHeap()
    xs = [rnd.randint(0, 100) for _ in range(200)]
    for x in xs:
        h.push(x, x)
    assert [h.pop() for _ in xs] == sorted(xs)
    q = ViolationQueue()
    q.push_violation("leve1", "leve"); q.push_violation("grav", "gravissima"); q.push_violation("leve2", "leve")
    assert [q.pop() for _ in range(3)] == ["grav", "leve1", "leve2"]


def test_plates():
    assert normalize("abc-1d23") == "ABC1D23" and is_valid_plate("ABC1D23")
    assert normalize("A8C1234") == "ABC1234"  # 8 virou B na posição de letra
    t = PlateTrie()
    for p in ["ABC1D23", "ABC1D24", "PQX9A88"]:
        t.insert(p)
    assert sorted(t.with_prefix("ABC")) == ["ABC1D23", "ABC1D24"]
    assert ("PQX9A88", 1) in t.fuzzy("PQX9A89", 1)


def test_bloom_no_false_negatives():
    bf = BloomFilter(1000, 0.01)
    keys = [f"k{i}" for i in range(1000)]
    for k in keys:
        bf.add(k)
    assert all(k in bf for k in keys)
    fp = sum(f"x{i}" in bf for i in range(10000)) / 10000
    assert fp < 0.03


def test_avl_balanced_and_range_sum():
    t = AVLTree()
    for i in range(1000):
        t.insert(i, f"inf{i}", points=3)
    assert t.height() <= 15
    assert t.sum_points(100, 199) == 300
    assert len(t.range(10, 19)) == 10


def test_fenwick():
    f = Fenwick(24)
    for h in [17, 17, 18, 19, 8]:
        f.add(h)
    assert f.range_sum(17, 19) == 4


def test_point_segment_distance():
    assert point_segment_distance((5, 3), (0, 0), (10, 0)) == 3
    assert point_segment_distance((13, 4), (0, 0), (10, 0)) == 5  # além da ponta: distância ao extremo
