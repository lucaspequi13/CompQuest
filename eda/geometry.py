"""Primitivas geométricas usadas pelas regras de infração.

- interseção de segmentos (cruzou a linha de retenção / linha contínua?)
- ponto em polígono por ray casting (está na faixa de pedestre / faixa exclusiva?)
- produto escalar entre vetores (anda na contramão?)
- distância ponto-segmento (está em cima da linha tracejada entre faixas?)
"""
from typing import Sequence, Tuple

Point = Tuple[float, float]


def cross(o: Point, a: Point, b: Point) -> float:
    """Produto vetorial (a-o) x (b-o). >0 anti-horário, <0 horário, 0 colinear."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True se o segmento p1-p2 cruza q1-q2. O(1)."""
    d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
    d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 and d2 and d3 and d4:
        return True
    if d1 == 0 and _on_segment(q1, p1, q2):
        return True
    if d2 == 0 and _on_segment(q1, p2, q2):
        return True
    if d3 == 0 and _on_segment(p1, q1, p2):
        return True
    if d4 == 0 and _on_segment(p1, q2, p2):
        return True
    return False


def side_of_line(a: Point, b: Point, p: Point) -> int:
    """Lado do ponto p em relação à reta a->b: 1, -1 ou 0."""
    c = cross(a, b, p)
    return (c > 0) - (c < 0)


def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    """Ray casting: conta quantas arestas um raio horizontal cruza. O(k)."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def dot(u: Point, v: Point) -> float:
    return u[0] * v[0] + u[1] * v[1]


def norm(u: Point) -> float:
    return (u[0] ** 2 + u[1] ** 2) ** 0.5


def cos_angle(u: Point, v: Point) -> float:
    nu, nv = norm(u), norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return dot(u, v) / (nu * nv)


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Distância do ponto p ao segmento ab (projeção limitada a [0, 1])."""
    ab = (b[0] - a[0], b[1] - a[1])
    ap = (p[0] - a[0], p[1] - a[1])
    L2 = dot(ab, ab)
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, dot(ap, ab) / L2))
    proj = (a[0] + t * ab[0], a[1] + t * ab[1])
    return norm((p[0] - proj[0], p[1] - proj[1]))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over Union de duas caixas (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def bottom_center(box: Sequence[float]) -> Point:
    """Ponto de contato do veículo com o chão (melhor que o centro da caixa)."""
    return ((box[0] + box[2]) / 2.0, float(box[3]))
