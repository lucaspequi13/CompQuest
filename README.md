# CompQuest — Fiscalização de trânsito por câmera

Software para câmeras urbanas de Goiânia que detecta infrações de trânsito, monta o pacote de
evidência e envia a um Detran simulado com pontuação educativa. Tema do campeonato:
**estruturas de dados e algoritmos** — só a detecção é rede neural; todo o resto está em `eda/`, feito do zero.

## Começar

```bash
bash setup.sh                       # cria .venv, instala tudo, roda os testes, prepara o VisDrone
source .venv/bin/activate
python scripts/train.py --quick     # confere se o treino roda (MPS no Mac)
```

## Pipeline

```
Câmera → [RingBuffer] → Detector YOLO → Tracker [Kalman + Húngaro] → Regras [geometria, grafo, FSM]
      → Evidência [hash encadeado] → Fila [MinHeap] → Detran sim [Bloom + AVL + Fenwick] → Dashboard
```

| Pasta | O que tem |
| --- | --- |
| `eda/` | ring_buffer, geometry, hungarian, kalman, quadtree, road_graph (BFS/Dijkstra), priority_heap, plate_trie (+Levenshtein), bloom_filter, avl_tree (soma por intervalo), segment_tree (Fenwick) |
| `perception/` | detector (YOLO), tracker (SORT próprio), traffic_light (HSV), blinker (série temporal), plate_reader (stub) |
| `rules/` | engine + `violations/checks.py`: sinal vermelho, contramão, faixa de pedestre, faixa de ônibus, linha contínua, conversão proibida, "gato" (canteiro e ré); `state_machine.py` para troca de faixa |
| `evidence/` | clipe + frame + JSON + SHA-256 encadeado |
| `detran_sim/` | regra de advertência, 3 leves → multa, limite de pontos; API FastAPI |
| `dashboard/` | Streamlit |
| `benchmarks/` | nossa estrutura vs. abordagem ingênua |
| `data/scenes/` | JSON de cada câmera (linhas, polígonos, grafo da via) — criar com `scripts/scene_editor.py` |

## Comandos

```bash
python -m pytest -q tests                                   # 18 testes
python benchmarks/bench_eda.py                              # números para os slides
python scripts/prepare_visdrone.py                          # converte VisDrone → YOLO
python scripts/train.py --model models/pretrained/yolo11s.pt --epochs 50
python scripts/run_pipeline.py --video data/videos/car-detection.mp4 \
    --scene data/scenes/exemplo_car_detection.json \
    --weights runs/train/visdrone_vehicles/weights/best.pt --light red
streamlit run dashboard/app.py -- --run runs/demo
uvicorn detran_sim.api:app --reload                         # API do Detran simulado
python scripts/scene_editor.py --video <video> --out data/scenes/<camera>.json
python scripts/train.py --resume runs/train/visdrone_vehicles/weights/last.pt   # se o treino parar
```

A troca de faixa sem seta (Art. 196) só é avaliada se a cena tiver `lanes` (ids das zonas que são faixas)
e `lane_lines` (linhas tracejadas entre elas) — tecla `d` no `scene_editor.py`. O `run_pipeline.py`
usa `models/blinker_rf.joblib` para a seta se existir; senão, o baseline heurístico.

## Benchmarks (medidos)

| Operação | Ingênuo | Nossa estrutura | Ganho |
| --- | --- | --- | --- |
| Busca espacial, 20 mil pontos, 2 mil consultas | lista: 2502 ms | quadtree: 110 ms | 23x |
| Soma de pontos em janela, 50 mil eventos | varredura: 5437 ms | AVL: 10,7 ms | 508x |
| Fila de prioridade, 20 mil infrações | lista + sort: 1016 ms | heap: 43 ms | 23x |

## Status / pendências

Contexto completo, decisões e "onde paramos": ver **CLAUDE.md**.


- **Fine-tune VisDrone (yolo11s, 50 épocas) em andamento** — log em `runs/logs/train_visdrone.log`. ~15 min/época no M5.
  AMP desligado no MPS (dava loss NaN). O COCO puro não reconhece carro visto de cima.
- **Vídeos de exemplo são de estacionamento**; a cena `exemplo_car_detection.json` é fictícia. Falta vídeo real de cruzamento de Goiânia.
- **Leitura de placa**: falta detector de placa BR (ver `docs/DATASETS.md`).
- **Seta**: 72 sequências do WIBAM extraídas; falta rotular (`label_blinkers.py`) e treinar.
- **Semáforo**: baseline por cor (HSV) na ROI da cena; BDD100K para algo robusto.
- CTB: pontos, multas, advertência (art. 267) e suspensão 20/30/40 (art. 261, I) conferidos. Demais artigos: ver CLAUDE.md.

Datasets: ver `docs/DATASETS.md`. Planejamento completo: doc "CompQuest — Fiscalização de Trânsito por Câmeras".
