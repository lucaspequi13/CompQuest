"""Converte o UA-DETRAC (cópia do Zenodo 14030107, já em YOLO) para as nossas 7 classes.

python scripts/prepare_uadetrac.py  (lê data/raw/UA-DETRAC.zip, não extrai o zip inteiro)
Saída: data/processed/uadetrac/{images,labels}/{train,val,test}

O zip tem 4 classes, conferidas visualmente: 0 car, 1 bus, 2 van, 3 others (caminhões).
É vídeo a 25 fps (câmera em viaduto, Pequim/Tianjin, dia/noite/chuva, 960x540): frames vizinhos
são quase iguais, então pegamos 1 a cada N frames.
- train: sequências MVI_2xxxx/MVI_4xxxx/... do split train, 1 a cada --train-stride
- val / test: split "val" do zip (sequências que não aparecem no treino), frames intercalados
"""
import argparse
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(ROOT, "data", "raw", "UA-DETRAC.zip")
OUT = os.path.join(ROOT, "data", "processed", "uadetrac")

# UA-DETRAC -> nossas classes (0 pedestrian, 1 bicycle, 2 car, 3 van, 4 truck, 5 bus, 6 motor)
MAP = {0: 2, 1: 5, 2: 3, 3: 4}


def frame_index(name: str) -> int:
    # MVI_20011__img00001.txt -> 1
    return int(name.rsplit("img", 1)[1].split(".")[0])


def export(z, members, split, keep):
    img_out = os.path.join(OUT, "images", split)
    lbl_out = os.path.join(OUT, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    n = 0
    for lbl in members:
        stem = lbl.split("/")[-1][:-4]
        if not keep(frame_index(stem)):
            continue
        img = lbl.replace("/labels/", "/images/")[:-4] + ".jpg"
        if img not in z.NameToInfo:
            continue
        lines = []
        for row in z.read(lbl).decode().splitlines():
            p = row.split()
            if len(p) == 5 and int(p[0]) in MAP:
                lines.append(" ".join([str(MAP[int(p[0])])] + p[1:]))
        with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        dst = os.path.join(img_out, stem + ".jpg")
        if not os.path.exists(dst):
            with open(dst, "wb") as f:
                f.write(z.read(img))
        n += 1
    print(f"[ok] {split}: {n} imagens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ZIP)
    ap.add_argument("--train-stride", type=int, default=5)
    ap.add_argument("--val-stride", type=int, default=10)
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    names = z.namelist()
    lbl = {s: sorted(n for n in names if n.startswith(f"UA-DETRAC/labels/{s}/") and n.endswith(".txt"))
           for s in ("train", "val")}
    ts, vs = args.train_stride, args.val_stride
    export(z, lbl["train"], "train", lambda i: i % ts == 0)
    export(z, lbl["val"], "val", lambda i: i % vs == 0)
    export(z, lbl["val"], "test", lambda i: i % vs == vs // 2)
    print("Agora: python scripts/train.py --data configs/uadetrac_mix.yaml "
          "--model runs/train/visdrone_vehicles/weights/best.pt --name uadetrac_ft")


if __name__ == "__main__":
    main()
