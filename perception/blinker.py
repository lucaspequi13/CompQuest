"""Detector de seta (pisca) por série temporal de brilho — baseline sem rede neural.

Ideia: recortar a região da lanterna traseira (ou dianteira) do veículo,
guardar o brilho médio numa janela deslizante e contar transições
liga/desliga. Pisca ≈ 1–2 Hz → em 30 fps, ~2 a 4 ciclos por 2 s.

TODO (fase 4): trocar o recorte heurístico por detecção de lanterna treinada no TLD.
"""
from collections import deque

import numpy as np


class BlinkerDetector:
    def __init__(self, fps: float = 30.0, window_s: float = 2.0, min_cycles: int = 2):
        self.fps = fps
        self.series = deque(maxlen=int(fps * window_s))
        self.min_cycles = min_cycles

    def push(self, crop_bgr) -> None:
        if crop_bgr is None or crop_bgr.size == 0:
            return
        # canal R pesado: luz de seta é âmbar
        r = crop_bgr[..., 2].astype(float)
        g = crop_bgr[..., 1].astype(float)
        self.series.append(float(np.mean(0.7 * r + 0.3 * g)))

    def is_blinking(self) -> bool:
        if len(self.series) < self.series.maxlen // 2:
            return False
        s = np.array(self.series)
        thr = (s.max() + s.min()) / 2
        if s.max() - s.min() < 25:  # pouca variação = luz parada
            return False
        on = s > thr
        transitions = int(np.sum(on[1:] != on[:-1]))
        cycles = transitions / 2
        freq = cycles / (len(s) / self.fps)
        return cycles >= self.min_cycles and 0.8 <= freq <= 2.5


def tail_light_rois(box, side: str):
    """Recorte aproximado das lanternas a partir da caixa do carro (câmera vendo a traseira)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    y_top, y_bot = y1 + 0.35 * h, y1 + 0.65 * h
    if side == "left":
        return (x1, y_top, x1 + 0.25 * w, y_bot)
    return (x2 - 0.25 * w, y_top, x2, y_bot)


class BlinkerClassifier:
    """Seta por veículo no pipeline ao vivo.

    Guarda os últimos `win_s` segundos de recortes de cada track (deque) e, a cada `step_s`,
    roda o RandomForest de scripts/train_blinker.py sobre as features FFT da janela.
    Sem modelo treinado, cai no BlinkerDetector heurístico (só diz "alerta"/"nenhuma",
    porque o baseline não sabe o lado).
    """

    def __init__(self, fps: float, model_path: str = "models/blinker_rf.joblib",
                 win_s: float = 1.5, step_s: float = 0.25, crop: int = 256):
        import os
        self.fps, self.crop = fps, crop
        self.win, self.step = int(win_s * fps), max(1, int(step_s * fps))
        self.model = None
        if model_path and os.path.exists(model_path):
            import joblib
            self.model = joblib.load(model_path)
        self.buf, self.label, self.since, self.fallback = {}, {}, {}, {}

    def update(self, track_id: int, frame, box) -> str:
        import cv2
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        pad = int(0.1 * max(x2 - x1, y2 - y1))
        c = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if c.size == 0:
            return self.label.get(track_id, "nenhuma")
        c = cv2.resize(c, (self.crop, self.crop))
        if self.model is None:
            det = self.fallback.setdefault(track_id, BlinkerDetector(self.fps))
            det.push(c)
            self.label[track_id] = "alerta" if det.is_blinking() else "nenhuma"
            return self.label[track_id]
        q = self.buf.setdefault(track_id, deque(maxlen=self.win))
        q.append(c)
        self.since[track_id] = self.since.get(track_id, 0) + 1
        if len(q) >= int(0.75 * self.win) and self.since[track_id] >= self.step:
            from perception.blinker_features import sequence_features
            self.since[track_id] = 0
            self.label[track_id] = str(self.model.predict([sequence_features(list(q), self.fps)])[0])
        return self.label.get(track_id, "nenhuma")

    def forget(self, alive_ids) -> None:
        """Libera memória dos tracks que sumiram."""
        for d in (self.buf, self.label, self.since, self.fallback):
            for k in [k for k in d if k not in alive_ids]:
                del d[k]
