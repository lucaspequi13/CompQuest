"""Roda todas as checagens de infração sobre os tracks de um frame."""
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List

from eda.road_graph import RoadGraph
from rules.types import Violation
from rules.violations.checks import ALL_CHECKS


@dataclass
class Context:
    scene: dict
    graph: RoadGraph
    frame_idx: int = 0
    timestamp: float = 0.0
    light: str = "unknown"
    hour: int = 12
    make: Callable = None


class RuleEngine:
    def __init__(self, scene_path: str, checks=None):
        with open(scene_path) as f:
            self.scene = json.load(f)
        self.graph = RoadGraph.from_scene(self.scene)
        self.checks = checks or ALL_CHECKS
        self.camera_id = self.scene.get("camera_id", "cam0")

    def evaluate(self, tracks, frame_idx: int, timestamp: float, light: str = "unknown") -> List[Violation]:
        ctx = Context(self.scene, self.graph, frame_idx, timestamp, light, datetime.fromtimestamp(timestamp).hour)
        ctx.make = lambda typ, sev, ctb, t: Violation(typ, sev, ctb, t.id, frame_idx, timestamp, self.camera_id, t.point)
        out = []
        for t in tracks:
            for check in self.checks:
                v = check(t, ctx)
                if v:
                    out.append(v)
        return out
