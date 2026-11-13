"""Máquina de estados de faixa por veículo.

Estados: NA_FAIXA -> CRUZANDO -> NOVA_FAIXA
Se chegou em NOVA_FAIXA sem seta ativa durante CRUZANDO -> infração (art. 196).
"""
from enum import Enum, auto


class LaneState(Enum):
    NA_FAIXA = auto()
    CRUZANDO = auto()


class LaneChangeFSM:
    def __init__(self, lane: str):
        self.state = LaneState.NA_FAIXA
        self.lane = lane
        self.signal_seen = False
        self.events = []

    def step(self, current_lane: str, blinking: bool):
        """current_lane=None quando o ponto está entre faixas (sobre a linha)."""
        if self.state == LaneState.NA_FAIXA:
            if current_lane is None:
                self.state = LaneState.CRUZANDO
                self.signal_seen = blinking
            elif current_lane != self.lane:  # pulou direto (frame perdido)
                ev = ("lane_change", self.lane, current_lane, blinking)
                self.lane = current_lane
                return ev
        elif self.state == LaneState.CRUZANDO:
            self.signal_seen |= blinking
            if current_lane is not None:
                self.state = LaneState.NA_FAIXA
                if current_lane != self.lane:
                    ev = ("lane_change", self.lane, current_lane, self.signal_seen)
                    self.lane = current_lane
                    return ev
        return None
