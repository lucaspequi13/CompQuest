"""Fine-tune do YOLO11 para veículos em câmera alta.

Teste rápido (≈ minutos, confere se está tudo ok):
    python scripts/train.py --quick
Treino de verdade no Mac (Apple Silicon usa "mps"):
    python scripts/train.py --model models/pretrained/yolo11s.pt --epochs 50
Se parar no meio (Mac dormiu, etc.):
    python scripts/train.py --resume runs/train/visdrone_vehicles/weights/last.pt
Pesos finais: runs/train/<nome>/weights/best.pt  -> usar em run_pipeline.py --weights
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def absolutize(yaml_path: str) -> str:
    """O Ultralytics resolve 'path' relativo a partir da pasta de datasets dele,
    não do YAML. Reescreve com caminho absoluto em runs/_data/."""
    import yaml
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg.get("path", ".")
    if not os.path.isabs(p):
        cfg["path"] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(yaml_path)), p))
    out_dir = os.path.join(ROOT, "runs", "_data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(yaml_path))
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return out


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "configs", "visdrone_vehicles.yaml"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "pretrained", "yolo11n.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="visdrone_vehicles")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4, help="processos de CPU carregando imagens")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--quick", action="store_true", help="1 época em 10%% dos dados, só para validar o setup")
    ap.add_argument("--resume", help="continuar um treino interrompido: caminho do last.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = absolutize(args.data)
    device = args.device or pick_device()
    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        model.val(data=data_yaml, split="test", device=device,
                  project=os.path.join(ROOT, "runs", "val"), name=args.name)
        return
    model = YOLO(args.model)
    kw = dict(
        data=data_yaml,
        epochs=1 if args.quick else args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=os.path.join(ROOT, "runs", "train"),
        name=args.name + ("_quick" if args.quick else ""),
        fraction=0.1 if args.quick else 1.0,
        patience=args.patience,
        workers=args.workers,
        plots=True,
        amp=device != "mps",  # fp16 no MPS gera loss NaN no meio da 1ª época (yolo11s, torch 2.8)
    )
    print("Treinando com:", kw)
    model.train(**kw)
    metrics = model.val(data=data_yaml, split="test" if not args.quick else "val", device=device,
                        project=os.path.join(ROOT, "runs", "val"), name=kw["name"])
    print("mAP50-95:", metrics.box.map, "mAP50:", metrics.box.map50)


if __name__ == "__main__":
    main()
