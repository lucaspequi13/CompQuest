"""Prepara um vídeo para o pipeline: reduz resolução, limita fps, corta trecho, gira vídeo deitado.

python scripts/prepare_video.py ENTRADA.mp4 SAIDA.mp4 [--max-side 1280] [--fps 30] [--start 0] [--end 0] [--rotate 0]
  --max-side  maior lado em pixels (4K a 60 fps deixa o YOLO lento e não ganha nada na demo)
  --fps       fps máximo (pula frames; 60 -> 30)
  --start/--end  trecho em segundos (end 0 = até o fim)
  --rotate    90 / 180 / 270 (sentido horário), se o vídeo vier deitado
"""
import argparse

import cv2

ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.src)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps_in / a.fps))
    fps_out = fps_in / step
    f0 = int(a.start * fps_in)
    f1 = int(a.end * fps_in) if a.end else n
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    wr, idx, written = None, f0, 0
    while idx < f1:
        ok, im = cap.read()
        if not ok:
            break
        if (idx - f0) % step == 0:
            if a.rotate:
                im = cv2.rotate(im, ROT[a.rotate])
            h, w = im.shape[:2]
            s = min(1.0, a.max_side / max(h, w))
            if s < 1.0:
                im = cv2.resize(im, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
            if wr is None:
                wr = cv2.VideoWriter(a.dst, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (im.shape[1], im.shape[0]))
            wr.write(im)
            written += 1
        idx += 1
    if wr:
        wr.release()
    print(f"{a.dst}: {written} frames, {fps_out:.1f} fps, {written / fps_out:.1f} s")


if __name__ == "__main__":
    main()
