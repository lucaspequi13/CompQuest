"""Grafo dirigido da via: faixas/zonas são nós, manobras permitidas são arestas.

- contramão / conversão proibida = o veículo foi de A para B sem aresta A->B
- BFS valida a sequência inteira de zonas percorrida por um track
- Dijkstra serve para o grafo da CIDADE (câmeras como nós, tempo de trajeto
  como peso): placa vista em 2 câmeras mais rápido que o caminho mínimo = suspeita de placa clonada
"""
import heapq
from collections import defaultdict, deque
from typing import Dict, Hashable, List, Optional, Tuple

Node = Hashable


class RoadGraph:
    def __init__(self):
        self.adj: Dict[Node, Dict[Node, float]] = defaultdict(dict)
        self.direction: Dict[Node, Tuple[float, float]] = {}  # vetor de sentido da faixa na imagem

    def add_zone(self, zone: Node, direction: Optional[Tuple[float, float]] = None) -> None:
        self.adj.setdefault(zone, {})
        if direction is not None:
            self.direction[zone] = direction

    def allow(self, a: Node, b: Node, weight: float = 1.0) -> None:
        self.add_zone(a)
        self.add_zone(b)
        self.adj[a][b] = weight

    def is_allowed(self, a: Node, b: Node) -> bool:
        """O(1) com dicionário de adjacência."""
        return a == b or b in self.adj.get(a, {})

    def first_illegal_move(self, path: List[Node]) -> Optional[Tuple[Node, Node]]:
        """Primeira transição proibida numa sequência de zonas (sem repetições consecutivas)."""
        for a, b in zip(path, path[1:]):
            if not self.is_allowed(a, b):
                return a, b
        return None

    def reachable(self, src: Node, dst: Node) -> bool:
        """BFS. O(V + E)."""
        seen, q = {src}, deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                return True
            for v in self.adj.get(u, {}):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return False

    def shortest_path(self, src: Node, dst: Node) -> Tuple[float, List[Node]]:
        """Dijkstra com heap. O(E log V). Retorna (custo, caminho)."""
        dist = {src: 0.0}
        prev: Dict[Node, Node] = {}
        pq = [(0.0, 0, src)]
        tie = 1
        while pq:
            d, _, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, float("inf")):
                continue
            for v, w in self.adj.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, tie, v))
                    tie += 1
        if dst not in dist:
            return float("inf"), []
        path, cur = [dst], dst
        while cur != src:
            cur = prev[cur]
            path.append(cur)
        return dist[dst], path[::-1]

    @classmethod
    def from_scene(cls, scene: dict) -> "RoadGraph":
        g = cls()
        for z in scene.get("zones", []):
            g.add_zone(z["id"], tuple(z["direction"]) if z.get("direction") else None)
        for a, b in scene.get("allowed_moves", []):
            g.allow(a, b)
        return g
