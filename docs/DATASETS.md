# Datasets — o que já está aqui e o que falta

Daqui só foi possível baixar o que está hospedado no GitHub. O resto exige formulário,
login ou um site que não estava liberado: **baixe manualmente no navegador** e coloque em `data/raw/`.

## Já baixado

| Arquivo | Onde | Uso |
| --- | --- | --- |
| VisDrone2019-DET train/val/test-dev (~1,9 GB) | `data/raw/` → convertido em `data/processed/visdrone` | Treino do detector de veículos em câmera alta (7 classes) |
| COCO128 | `data/raw/coco128.zip` | Smoke test de treino |
| OpenALPR benchmarks (inclui placas BR) | `data/raw/openalpr-benchmarks.zip` | Testar OCR de placa (pasta `endtoend/br`) |
| YOLO11 n / s / m (COCO) | `models/pretrained/` | Ponto de partida do fine-tune |
| 2 vídeos de exemplo (estacionamento, 768x432) | `data/videos/` | Testar o pipeline |

## Falta baixar (manual)

| Dataset | Para quê | Como conseguir |
| --- | --- | --- |
| UA-DETRAC | Detecção + tracking em câmera de trânsito (melhor base para o nosso caso) | Kaggle: buscar "UA-DETRAC" (há cópias em formato YOLO) |
| MIO-TCD | Detecção/classificação em câmera fixa (11 classes) | https://tcd.miovision.com/challenge/dataset.html |
| RodoSol-ALPR | Placas BR antigas e Mercosul (20 mil imagens) | Formulário em https://github.com/raysonlaroca/rodosol-alpr-dataset |
| UFPR-ALPR | Placas BR em cenário urbano | Formulário em https://web.inf.ufpr.br/vri/databases/ufpr-alpr/ |
| TLD | Seta / luz de freio | https://huggingface.co/datasets/ChaiJohn/TLD |
| BDD100K | Semáforo (estado) e tipos de linha | http://bdd-data.berkeley.edu (cadastro) |
| CityFlow (AI City) | Multi-câmera / ReID | https://www.aicitychallenge.org (cadastro) |
| BrnoCompSpeed | Velocidade | https://github.com/JakubSochor/BrnoCompSpeed (~195 GB, pedir por e-mail) |
| Roboflow Universe | Bases pequenas prontas (capacete, semáforo, placa BR) | https://universe.roboflow.com — exportar em "YOLOv11" |

## Recomendado: dataset próprio de Goiânia

10–20 min de vídeo de cruzamentos (de uma passarela, câmera parada), anotar no CVAT ou Roboflow.
É o único teste que mostra o modelo funcionando no trânsito daqui. Borrar placas e rostos no material público.

## Formato YOLO (para qualquer dataset novo)

```
data/processed/<nome>/images/{train,val}/*.jpg
data/processed/<nome>/labels/{train,val}/*.txt   # classe cx cy w h (normalizados 0–1)
```
Criar `configs/<nome>.yaml` igual ao `visdrone_vehicles.yaml` e treinar com `--data configs/<nome>.yaml`.
