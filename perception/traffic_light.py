"""Estado do semáforo por cor dominante (HSV) na região marcada na cena.

Baseline sem treino. Substituir por um classificador treinado (BDD100K) depois.
"""
import cv2
import numpy as np


def light_state(frame, roi) -> str:
    """roi = [x1, y1, x2, y2] do semáforo na imagem. Retorna 'red', 'yellow', 'green' ou 'unknown'."""
    x1, y1, x2, y2 = map(int, roi)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bright = hsv[..., 2] > 150
    sat = hsv[..., 1] > 100
    mask = bright & sat
    if mask.sum() < 5:
        return "unknown"
    h = hsv[..., 0][mask]
    red = np.sum((h < 10) | (h > 165))
    yellow = np.sum((h >= 15) & (h < 35))
    green = np.sum((h >= 40) & (h < 95))
    return ["red", "yellow", "green"][int(np.argmax([red, yellow, green]))]
