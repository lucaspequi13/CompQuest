"""Quadtree para busca espacial: "quais veículos estão nesta região?".

Inserção e consulta O(log n) em média, contra O(n) de varrer uma lista.
Ver benchmarks/bench_spatial.py para a comparação.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def intersects(self, o: "Rect") -> bool:
        return not (o.x >= self.x + self.w or o.x + o.w <= self.x or o.y >= self.y + self.h or o.y + o.h <= self.y)


@dataclass
class QuadTree:
    boundary: Rect
    capacity: int = 8
    max_depth: int = 10
    depth: int = 0
    points: List[Tuple[float, float, Any]] = field(default_factory=list)
    children: Optional[List["QuadTree"]] = None

    def insert(self, px: float, py: float, data: Any = None) -> bool:
        if not self.boundary.contains(px, py):
            return False
        if self.children is None:
            if len(self.points) < self.capacity or self.depth >= self.max_depth:
                self.points.append((px, py, data))
                return True
            self._subdivide()
        return any(c.insert(px, py, data) for c in self.children)  # type: ignore[union-attr]

    def _subdivide(self) -> None:
        b, hw, hh = self.boundary, self.boundary.w / 2, self.boundary.h / 2
        self.children = [
            QuadTree(Rect(b.x, b.y, hw, hh), self.capacity, self.max_depth, self.depth + 1),
            QuadTree(Rect(b.x + hw, b.y, hw, hh), self.capacity, self.max_depth, self.depth + 1),
            QuadTree(Rect(b.x, b.y + hh, hw, hh), self.capacity, self.max_depth, self.depth + 1),
            QuadTree(Rect(b.x + hw, b.y + hh, hw, hh), self.capacity, self.max_depth, self.depth + 1),
        ]
        for px, py, d in self.points:
            any(c.insert(px, py, d) for c in self.children)
        self.points = []

    def query(self, area: Rect, found: Optional[list] = None) -> list:
        found = [] if found is None else found
        if not self.boundary.intersects(area):
            return found
        for px, py, d in self.points:
            if area.contains(px, py):
                found.append((px, py, d))
        if self.children:
            for c in self.children:
                c.query(area, found)
        return found

    def query_radius(self, cx: float, cy: float, r: float) -> list:
        cand = self.query(Rect(cx - r, cy - r, 2 * r, 2 * r))
        return [p for p in cand if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= r * r]
