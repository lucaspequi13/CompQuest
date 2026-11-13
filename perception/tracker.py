"""Tracker multi-objeto estilo SORT, feito com as nossas estruturas:
Kalman (predição) + Húngaro (associação por 1 - IoU) + deque (trajetória).
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

from eda.geometry import bottom_center, iou
from eda.hungarian import hungarian
from eda.kalman import KalmanCV


@dataclass
class Track:
    id: int
    box: tuple
    cls: str
    kf: KalmanCV
    hits: int = 1
    misses: int = 0
    trail: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=90))
    zones: List[str] = field(default_factory=list)  # sequência de zonas visitadas (sem repetição)
    state: Dict = field(default_factory=dict)        # usado pelas regras (máquina de estados)

    @property
    def point(self):
        return bottom_center(self.box)

    @property
    def velocity(self):
        return self.kf.velocity


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 15, min_hits: int = 3):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections) -> List[Track]:
        for t in self.tracks:
            t.kf.predict()

        matched_t, matched_d = set(), set()
        if self.tracks and detections:
            cost = [[1 - iou(t.box, d.box) for d in detections] for t in self.tracks]
            for ti, di in hungarian(cost):
                if 1 - cost[ti][di] >= self.iou_threshold:
                    t, d = self.tracks[ti], detections[di]
                    t.box, t.cls = d.box, d.cls
                    t.kf.update(*t.point)
                    t.trail.append(t.point)
                    t.hits += 1
                    t.misses = 0
                    matched_t.add(ti)
                    matched_d.add(di)

        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.misses += 1

        for di, d in enumerate(detections):
            if di not in matched_d:
                cx, cy = bottom_center(d.box)
                t = Track(self._next_id, d.box, d.cls, KalmanCV(cx, cy))
                t.trail.append((cx, cy))
                self.tracks.append(t)
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses == 0]
