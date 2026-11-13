"""Features de seta a partir de uma sequência de recortes do veículo.

Ideia (EDA + sinais): pisca = luz âmbar que liga/desliga a ~1–2 Hz.
1. Divide o recorte em metade esquerda e direita (a seta acende de um lado só).
2. Em cada metade, mede o brilho âmbar por frame -> série temporal.
3. Janela deslizante + FFT: energia na banda 0,8–2,5 Hz vs. resto, nº de transições,
   e diferença esquerda-direita.
4. Mapa por pixel (64x64): energia na banda / energia total x amplitude, por metade.
   A lanterna é pequena e some na média da metade; o máximo do mapa por pixel acha ela.
Resultado: vetor pequeno (22 números) -> classificador leve (RandomForest).
"""
import numpy as np

BAND = (0.8, 2.5)  # Hz típico de pisca


def amber_score(crop_bgr) -> np.ndarray:
    """Mapa de 'quão âmbar/brilhante' é cada pixel."""
    c = crop_bgr.astype(np.float32)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    amber = np.clip(r - b, 0, None) * (g > 0.35 * r) * (r > 120)
    return amber


def side_series(crops):
    """Duas séries (esquerda, direita): média do top-5% de pixels âmbar em cada metade."""
    left, right = [], []
    for c in crops:
        a = amber_score(c)
        h, w = a.shape
        for half, out in ((a[:, : w // 2], left), (a[:, w // 2:], right)):
            v = np.sort(half.ravel())
            out.append(float(v[int(0.95 * len(v)):].mean()))
    return np.array(left), np.array(right)


def series_features(s: np.ndarray, fps: float):
    s = s - s.mean()
    n = len(s)
    if n < 8 or np.allclose(s, 0):
        return [0.0] * 7
    spec = np.abs(np.fft.rfft(s * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / fps)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    total = spec[1:].sum() + 1e-9
    band_ratio = spec[band].sum() / total
    peak_f = freqs[1:][np.argmax(spec[1:])] if n > 2 else 0.0
    thr = 0.0
    on = s > thr
    transitions = int(np.sum(on[1:] != on[:-1]))
    rate = transitions / 2 / (n / fps)
    return [float(band_ratio), float(peak_f), float(rate), float(s.std()),
            float(np.ptp(s)), float(spec[band].max() if band.any() else 0), float(transitions)]


def pixel_band_map(crops, fps: float, size: int = 64) -> np.ndarray:
    """Mapa (size x size): quanto cada pixel 'pisca' na banda do pisca (razão na banda x amplitude)."""
    import cv2
    a = np.stack([amber_score(cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)) for c in crops])
    n = len(a)
    if n < 8:
        return np.zeros((size, size), np.float32)
    a = a - a.mean(0)
    spec = np.abs(np.fft.rfft(a * np.hanning(n)[:, None, None], axis=0)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / fps)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    e = spec[band].sum(0)
    m = e / (spec[1:].sum(0) + 1e-9) * np.sqrt(e)
    return cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)


def pixel_features(crops, fps: float):
    """Por metade: máximo do mapa, altura relativa do máximo; + razão esq/dir."""
    m = pixel_band_map(crops, fps)
    h, w = m.shape
    out = []
    for half in (m[:, : w // 2], m[:, w // 2:]):
        y, _ = np.unravel_index(half.argmax(), half.shape)
        out += [float(half.max()), y / h]
    out.append(np.log1p(out[0]) - np.log1p(out[2]))
    return out


def sequence_features(crops, fps: float):
    l, r = side_series(crops)
    fl, fr = series_features(l, fps), series_features(r, fps)
    diff = [fl[0] - fr[0], fl[3] - fr[3], float(np.corrcoef(l, r)[0, 1]) if l.std() and r.std() else 0.0]
    return np.array(fl + fr + diff + pixel_features(crops, fps), dtype=np.float32)


def windows(crops, fps: float, win_s: float = 1.5, step_s: float = 0.25):
    """Janelas deslizantes sobre a sequência (para inferência ao vivo)."""
    w, st = int(win_s * fps), max(1, int(step_s * fps))
    for i in range(0, max(1, len(crops) - w + 1), st):
        yield i, crops[i:i + w]


LABELS = ["nenhuma", "esquerda", "direita", "alerta"]
