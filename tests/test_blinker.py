"""Features de seta com recortes sintéticos: pisca âmbar de 1,5 Hz só na metade esquerda."""
import numpy as np

from perception.blinker_features import sequence_features, side_series, series_features


def _crops(fps=25, seconds=2.0, hz=1.5, side="left"):
    out = []
    for i in range(int(fps * seconds)):
        c = np.full((64, 64, 3), 90, np.uint8)
        if np.sin(2 * np.pi * hz * i / fps) > 0:
            x = slice(4, 14) if side == "left" else slice(50, 60)
            c[28:36, x] = (0, 170, 255)  # BGR âmbar
        out.append(c)
    return out


def test_blinking_side_has_band_energy():
    fps = 25
    left, right = side_series(_crops(fps))
    fl, fr = series_features(left, fps), series_features(right, fps)
    assert fl[0] > 0.6 and 0.8 <= fl[1] <= 2.5   # energia na banda do pisca, pico ~1,5 Hz
    assert fr[3] == 0                             # lado direito não muda
    assert sequence_features(_crops(fps), fps).shape == (22,)
