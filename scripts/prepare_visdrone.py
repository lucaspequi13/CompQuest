"""Converte o VisDrone2019-DET para formato YOLO, mantendo só as classes de trânsito.

python scripts/prepare_visdrone.py  (lê data/raw/VisDrone2019-DET-*.zip)
Saída: data/processed/visdrone/{images,labels}/{train,val,test}

Formato VisDrone: x,y,w,h,score,categoria,truncamento,oclusão
score 0 = região ignorada. Categorias 1..10:
1 pedestrian 2 people 3 bicycle 4 car 5 van 6 truck 7 tricycle 8 awning-tricycle 9 bus 10 motor
"""
import os
import shutil
import zipfile

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed", "visdrone")

CLASSES = ["pedestrian", "bicycle", "car", "van", "truck", "bus", "motor"]
MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 9: 5, 10: 6}  # triciclos ficam de fora

SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}


def convert(split: str, name: str) -> int:
    zpath = os.path.join(RAW, name + ".zip")
    if not os.path.exists(zpath):
        print(f"[pula] {zpath} não encontrado")
        return 0
    tmp = os.path.join(RAW, "_extract")
    src = os.path.join(tmp, name)
    if not os.path.isdir(src):
        print(f"[unzip] {name}")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
    img_out = os.path.join(OUT, "images", split)
    lbl_out = os.path.join(OUT, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    n = 0
    for fname in sorted(os.listdir(os.path.join(src, "annotations"))):
        stem = fname[:-4]
        img_src = os.path.join(src, "images", stem + ".jpg")
        if not os.path.exists(img_src):
            continue
        w, h = Image.open(img_src).size
        lines = []
        with open(os.path.join(src, "annotations", fname)) as f:
            for row in f:
                p = row.strip().rstrip(",").split(",")
                if len(p) < 6 or p[4] == "0":
                    continue
                cat = int(p[5])
                if cat not in MAP:
                    continue
                x, y, bw, bh = map(float, p[:4])
                if bw <= 1 or bh <= 1:
                    continue
                lines.append(f"{MAP[cat]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        dst = os.path.join(img_out, stem + ".jpg")
        if not os.path.exists(dst):
            shutil.move(img_src, dst)
        n += 1
    print(f"[ok] {split}: {n} imagens")
    return n


if __name__ == "__main__":
    for split, name in SPLITS.items():
        convert(split, name)
    print("Classes:", CLASSES)
    print("Agora: python scripts/train.py --data configs/visdrone_vehicles.yaml")
