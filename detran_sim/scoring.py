"""Simulação educativa de pontos e multas.

Regras:
1. Leve/média, sem outra infração nos últimos 12 meses -> ADVERTÊNCIA (CTB art. 267)
2. 3 leves acumuladas em 12 meses -> vira multa leve + 3 pontos (regra do time)
3. Grave/gravíssima -> multa + pontos direto
4. Suspensão (CTB art. 261, I, redação da Lei 14.071/2020): 20 pts com 2+ gravíssimas
   em 12 meses, 30 com 1, 40 sem nenhuma. Motorista profissional (EAR na CNH) é sempre 40
   — não modelado aqui porque a câmera não sabe quem tem EAR.

Estruturas: dict placa -> AVLTree(timestamp) com soma de pontos por subárvore,
Bloom filter para deduplicar e Fenwick por hora para o mapa de calor.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from eda.avl_tree import AVLTree
from eda.bloom_filter import BloomFilter, violation_key
from eda.segment_tree import Fenwick
from rules.types import FINES, POINTS

YEAR = 365 * 24 * 3600


@dataclass
class Driver:
    plate: str
    history: AVLTree = field(default_factory=AVLTree)        # pontos efetivamente aplicados
    all_events: AVLTree = field(default_factory=AVLTree)     # tudo, inclusive advertências
    pending_leves: int = 0


class DetranSim:
    def __init__(self):
        self.drivers: Dict[str, Driver] = {}
        self.dedup = BloomFilter(1_000_000, 0.001)
        self.by_hour: Dict[str, Fenwick] = defaultdict(lambda: Fenwick(24))
        self.log: List[dict] = []

    def _driver(self, plate: str) -> Driver:
        return self.drivers.setdefault(plate, Driver(plate))

    def receive(self, v: dict) -> dict:
        plate = v.get("plate") or f"TRACK-{v['camera_id']}-{v['track_id']}"
        ts = float(v["timestamp"])
        key = violation_key(plate, v["type"], int(ts // 300))
        if self.dedup.check_and_add(key):
            return {"status": "duplicada", "plate": plate}

        sev = v["severity"]
        d = self._driver(plate)
        recent = d.all_events.range(ts - YEAR, ts)
        d.all_events.insert(ts, v, 0)
        from datetime import datetime
        self.by_hour[v["camera_id"]].add(datetime.fromtimestamp(ts).hour)

        if sev in ("leve", "media") and not recent:
            decision = {"acao": "advertencia", "pontos": 0, "multa": 0.0}
        elif sev == "leve":
            d.pending_leves += 1
            if d.pending_leves >= 3:
                d.pending_leves = 0
                d.history.insert(ts, v, POINTS["leve"])
                decision = {"acao": "multa_por_acumulo_de_leves", "pontos": POINTS["leve"], "multa": FINES["leve"]}
            else:
                decision = {"acao": f"leve_registrada ({d.pending_leves}/3)", "pontos": 0, "multa": 0.0}
        else:
            d.history.insert(ts, v, POINTS[sev])
            decision = {"acao": "multa", "pontos": POINTS[sev], "multa": FINES[sev]}

        total = d.history.sum_points(ts - YEAR, ts)
        grav = sum(1 for _, e in d.history.range(ts - YEAR, ts) if e["severity"] == "gravissima")
        limit = 20 if grav >= 2 else 30 if grav == 1 else 40
        decision.update({"plate": plate, "pontos_12m": total, "limite": limit, "suspensa": total >= limit, "tipo": v["type"]})
        self.log.append(decision)
        return decision

    def hotspot(self, camera_id: str, h_from: int, h_to: int) -> int:
        return self.by_hour[camera_id].range_sum(h_from, h_to)
