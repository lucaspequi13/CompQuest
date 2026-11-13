from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

# Natureza da infração -> pontos na CNH (CTB art. 259)
POINTS = {"leve": 3, "media": 4, "grave": 5, "gravissima": 7}
# Valor base em R$ (CTB art. 258)
FINES = {"leve": 88.38, "media": 130.16, "grave": 195.23, "gravissima": 293.47}


@dataclass
class Violation:
    type: str
    severity: str  # leve | media | grave | gravissima
    ctb: str
    track_id: int
    frame_idx: int
    timestamp: float
    camera_id: str
    point: Tuple[float, float]
    plate: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def points(self) -> int:
        return POINTS[self.severity]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["points"] = self.points
        return d
