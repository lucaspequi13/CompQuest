"""Treino pequeno do classificador de seta (sem GPU, segundos).

python scripts/train_blinker.py
Lê data/processed/blinker/{seqs,labels.csv}, extrai features (FFT por lado),
aumenta os dados com janelas deslizantes e espelhamento (esq<->dir),
treina RandomForest com validação por sequência e salva models/blinker_rf.joblib.

Rótulo é da sequência inteira, mas a seta costuma piscar só num trecho. Nas sequências
com seta ficam só as janelas com mais "pisca" (metade de cima pelo máximo do mapa por pixel);
o resto é ambíguo e sai do treino (supervisão fraca, estilo multiple-instance).
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from perception.blinker_features import sequence_features, windows  # noqa: E402

WIN_S, MIN_S = 1.5, 1.0
PIX_L, PIX_R = 17, 19  # posições do máximo do mapa por pixel (esq/dir) no vetor

BASE = os.path.join(ROOT, "data", "processed", "blinker")
MIRROR = {"esquerda": "direita", "direita": "esquerda", "nenhuma": "nenhuma", "alerta": "alerta"}


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GroupKFold
    import joblib

    with open(os.path.join(BASE, "labels.csv")) as f:
        labels = {r["seq"]: r["label"] for r in csv.DictReader(f) if r["label"] != "descartar"}
    print("Rótulos:", Counter(labels.values()))

    X, y, groups = [], [], []
    for seq, lab in labels.items():
        d = os.path.join(BASE, "seqs", seq)
        crops = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "*.jpg")))]
        fps = json.load(open(os.path.join(d, "meta.json"))).get("fps", 25)
        for flip in (False, True):
            cs = [cv2.flip(c, 1) for c in crops] if flip else crops
            lb = MIRROR[lab] if flip else lab
            feats = [sequence_features(w, fps) for _, w in windows(cs, fps, win_s=WIN_S)
                     if len(w) >= int(MIN_S * fps)]
            if lab not in ("nenhuma", "descartar") and len(feats) > 1:
                score = [max(f[PIX_L], f[PIX_R]) for f in feats]
                feats = [f for f, s in zip(feats, score) if s >= np.median(score)]
            for f in feats:
                X.append(f)
                y.append(lb)
                groups.append(seq)  # mesma sequência nunca fica em treino e validação ao mesmo tempo
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    print(f"{len(X)} janelas de {len(set(groups))} sequências")

    n_splits = min(5, len(set(groups)))
    if n_splits >= 2:
        preds = np.empty_like(y)
        for tr, te in GroupKFold(n_splits).split(X, y, groups):
            m = RandomForestClassifier(300, class_weight="balanced", random_state=0).fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
        print(classification_report(y, preds, zero_division=0))
        # por sequência: tem seta se alguma janela (original ou espelhada) foi classificada com seta
        print("Por sequência (validação cruzada):")
        hit = {}
        for g, yt, yp in zip(groups, y, preds):
            hit.setdefault(g, [labels[g], False])[1] |= yp != "nenhuma"
        tp = sum(1 for lab, p in hit.values() if lab != "nenhuma" and p)
        fn = sum(1 for lab, p in hit.values() if lab != "nenhuma" and not p)
        fp = sum(1 for lab, p in hit.values() if lab == "nenhuma" and p)
        tn = sum(1 for lab, p in hit.values() if lab == "nenhuma" and not p)
        print(f"  seta achada {tp}/{tp + fn} · falso alarme {fp}/{fp + tn}")
        for g, (lab, p) in sorted(hit.items()):
            if lab != "nenhuma" or p:
                print(f"  {g:22s} rótulo={lab:9s} previu_seta={p}")

    model = RandomForestClassifier(300, class_weight="balanced", random_state=0).fit(X, y)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    out = os.path.join(ROOT, "models", "blinker_rf.joblib")
    joblib.dump(model, out)
    print("Modelo salvo em", out)


if __name__ == "__main__":
    main()
