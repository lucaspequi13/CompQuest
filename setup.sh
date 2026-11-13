#!/usr/bin/env bash
# Setup no Mac (Apple Silicon). Rodar uma vez:  bash setup.sh
set -e
cd "$(dirname "$0")"

PY=${PY:-python3}
echo ">> Python: $($PY --version)"
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ">> Testes das estruturas de dados e regras"
python -m pytest -q tests

echo ">> Conferindo datasets"
[ -d data/processed/visdrone/images/train ] || python scripts/prepare_visdrone.py

python - <<'EOF'
import torch
print("MPS (GPU do Mac) disponível:", torch.backends.mps.is_available())
EOF

echo
echo "Pronto. Próximos passos:"
echo "  source .venv/bin/activate"
echo "  python scripts/train.py --quick            # teste de treino (minutos)"
echo "  python scripts/train.py --model models/pretrained/yolo11s.pt --epochs 50"
echo "  python scripts/run_pipeline.py --video data/videos/car-detection.mp4 --scene data/scenes/exemplo_car_detection.json --light red"
echo "  streamlit run dashboard/app.py -- --run runs/demo"
