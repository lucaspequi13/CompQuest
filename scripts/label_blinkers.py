"""Ferramenta de rotulagem rápida das setas.

python scripts/label_blinkers.py
Mostra cada sequência em loop (ampliada). Teclas:
  n = nenhuma   e = esquerda   d = direita   a = alerta (pisca-alerta)
  x = descartar (não dá pra ver / recorte ruim)   b = voltar   q = sair (salva)
'Esquerda/direita' = do ponto de vista do MOTORISTA (se a câmera vê a traseira,
é o mesmo lado da imagem; se vê a frente, é o lado oposto).
Rótulos vão para data/processed/blinker/labels.csv (dá para parar e continuar depois).
"""
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from perception.blinker_features import side_series  # noqa: E402

BASE = os.path.join(ROOT, "data", "processed", "blinker")
KEYS = {ord("n"): "nenhuma", ord("e"): "esquerda", ord("d"): "direita", ord("a"): "alerta", ord("x"): "descartar"}


def load_labels(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["seq"]: r["label"] for r in csv.DictReader(f)}


def save_labels(path, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "label"])
        for k, v in sorted(labels.items()):
            w.writerow([k, v])


def plot_series(l, r, w=512, h=80):
    img = np.zeros((h, w, 3), np.uint8)
    for s, color in ((l, (0, 200, 255)), (r, (255, 120, 0))):
        if len(s) < 2 or np.ptp(s) == 0:
            continue
        ys = h - 5 - (s - s.min()) / np.ptp(s) * (h - 10)
        xs = np.linspace(0, w - 1, len(s))
        cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, color, 1)
    cv2.putText(img, "esq", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    cv2.putText(img, "dir", (40, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 120, 0), 1)
    return img


def main():
    seqs = sorted(glob.glob(os.path.join(BASE, "seqs", "*")))
    path = os.path.join(BASE, "labels.csv")
    labels = load_labels(path)
    todo = [s for s in seqs if os.path.basename(s) not in labels]
    print(f"{len(seqs)} sequências, {len(labels)} já rotuladas, {len(todo)} faltando")
    i = 0
    while 0 <= i < len(todo):
        d = todo[i]
        name = os.path.basename(d)
        files = sorted(glob.glob(os.path.join(d, "*.jpg")))
        crops = [cv2.imread(f) for f in files]
        fps = json.load(open(os.path.join(d, "meta.json"))).get("fps", 25)
        l, r = side_series(crops)
        graph = plot_series(l, r)
        delay = max(1, int(1000 / fps))
        k = -1
        while k == -1 or (k not in KEYS and k not in (ord("b"), ord("q"))):
            for c in crops:
                big = cv2.resize(c, (512, 512), interpolation=cv2.INTER_NEAREST)
                cv2.putText(big, f"{i + 1}/{len(todo)} {name}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("seta? n/e/d/a/x  b=voltar q=sair", np.vstack([big, graph]))
                k = cv2.waitKey(delay) & 0xFF
                if k != 255:
                    break
            else:
                k = -1
        if k == ord("q"):
            break
        if k == ord("b"):
            i = max(0, i - 1)
            labels.pop(os.path.basename(todo[i]), None)
            continue
        labels[name] = KEYS[k]
        save_labels(path, labels)
        i += 1
    save_labels(path, labels)
    cv2.destroyAllWindows()
    print("salvo em", path)


if __name__ == "__main__":
    main()
