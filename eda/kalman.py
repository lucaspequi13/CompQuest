"""Filtro de Kalman de velocidade constante para o centro da caixa.

Estado x = [cx, cy, vx, vy]. Prevê onde o veículo estará no próximo frame,
o que ajuda a manter o ID durante oclusões curtas. O(1) por veículo.
"""
import numpy as np


class KalmanCV:
    def __init__(self, cx: float, cy: float, q: float = 1.0, r: float = 10.0):
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.eye(4) * 100.0
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * q
        self.R = np.eye(2) * r

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, cx: float, cy: float) -> np.ndarray:
        z = np.array([cx, cy])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2]

    @property
    def velocity(self):
        return float(self.x[2]), float(self.x[3])
