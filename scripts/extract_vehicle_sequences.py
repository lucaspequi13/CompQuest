"""Gera sequências de recortes por veículo (entrada do treino de seta).

Roda detector + nosso tracker sobre um vídeo OU uma pasta de frames numerados
e salva, para cada track longo o bastante, os recortes do veículo frame a frame.

python scripts/extract_vehicle_sequences.py --frames data/raw/wibam_demo/frames/2 --fps 50 --prefix wibam_c2
python scripts/extract_vehicle_sequences.py --video data/videos/meu_cruzamento.mp4 --prefix goiania1

Saída: data/processed/blinker/seqs/<prefix>_t<id>/0000.jpg ... + meta.json
"""
import argparse
import glob
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from perception.detector import Detector  # noqa: E402
from perception.tracker import Tracker  # noqa: E402

VEHICLES = {"car", "truck", "bus", "van", "motorcycle", "motor"}


def frames_from(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f
    else:
        files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")), key=lambda p: int(os.path.basename(p)[:-4]))
        prev = None
        for p in files:
            n = int(os.path.basename(p)[:-4])
            if prev is not None and n - prev > 1:  # buraco = clipe novo
                yield None
            prev = n
            yield cv2.imread(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--frames")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--weights", default=os.path.join(ROOT, "models", "pretrained", "yolo11s.pt"))
    ap.add_argument("--min-len", type=int, default=20, help="mínimo de frames por sequência")
    ap.add_argument("--min-size", type=int, default=60, help="lado mínimo do recorte em px")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "processed", "blinker", "seqs"))
    args = ap.parse_args()
    assert args.video or args.frames, "passe --video ou --frames"

    det = Detector(args.weights, conf=0.4)
    trk = Tracker(min_hits=1, max_misses=5)
    seqs, clip = {}, 0

    def flush():
        nonlocal seqs
        saved = 0
        for tid, crops in seqs.items():
            if len(crops) < args.min_len:
                continue
            d = os.path.join(args.out, f"{args.prefix}_c{clip}_t{tid}")
            os.makedirs(d, exist_ok=True)
            for i, (img, box) in enumerate(crops):
                cv2.imwrite(os.path.join(d, f"{i:04d}.jpg"), img)
            json.dump({"fps": args.fps, "n": len(crops), "boxes": [b for _, b in crops],
                       "source": args.video or args.frames}, open(os.path.join(d, "meta.json"), "w"))
            saved += 1
        print(f"clipe {clip}: {saved} sequências salvas")
        seqs = {}

    for frame in frames_from(args):
        if frame is None:
            flush()
            clip += 1
            trk = Tracker(min_hits=1, max_misses=5)
            continue
        h, w = frame.shape[:2]
        dets = [d for d in det(frame) if d.cls in VEHICLES]
        for t in trk.update(dets):
            x1, y1, x2, y2 = map(int, t.box)
            if min(x2 - x1, y2 - y1) < args.min_size:
                continue
            pad = int(0.1 * max(x2 - x1, y2 - y1))
            c = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
            seqs.setdefault(t.id, []).append((cv2.resize(c, (args.crop, args.crop)), [x1, y1, x2, y2]))
    flush()


if __name__ == "__main__":
    main()
