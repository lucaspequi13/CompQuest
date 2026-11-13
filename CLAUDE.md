# CompQuest — Fiscalização de trânsito por câmera

Contexto completo do projeto para o Claude Code (e para o time). Atualizado em 13/11/2026.
**Responder em português informal, direto e conciso. Preferir tabelas e listas curtas a texto longo.**

---

## 1. O que é

- **Campeonato:** CompQuest (estilo hackathon). **Tema da equipe: Estruturas de Dados e Algoritmos (EDA).**
- **Ideia:** Goiânia tem muitas câmeras em ruas pavimentadas e a prefeitura quer ampliar (meta de ~20 mil câmeras; PPP de 10 mil; já houve identificação de criminosos por elas). Nossa proposta é um **software que roda nessas câmeras** e identifica infrações de trânsito — das pequenas (trocar de faixa sem seta) às grandes (contramão, avanço de sinal) —, monta a evidência (frames + clipe + placa) e **envia ao Detran**, que faz a parte dele.
- **Simulação educativa:** o objetivo é corrigir o trânsito aos poucos. Simulamos pontos e multas, com regra própria de acúmulo: **3 infrações leves → multa leve**.
- **Gancho do tema:** a rede neural (YOLO) só gera caixas por frame. Tudo que transforma caixas em infração é **EDA implementada do zero** (`eda/`). Esse é o diferencial da apresentação — **não trocar por bibliotecas prontas**.

## 2. Escopo e decisões (importante)

- **É uma DEMO, não um produto.** Não precisamos provar que o modelo é perfeito; precisamos mostrar que **quando a infração ocorre numa câmera limpa e bem posicionada, o sistema identifica**. Treinos pequenos/amadores são aceitáveis.
- **Só câmeras bem posicionadas:** fixas, alta resolução, ângulo que permita ver setas e manobras.
- **Prioridade de dados:** vídeos/frames **limpos e bem visíveis**, de preferência de países desenvolvidos (Alemanha, EUA, Austrália, Espanha). Filtrar e escolher os melhores.
- **NÃO usar datasets de câmera de painel (dashcam), ex.: TLD** — a perspectiva (de dentro do carro, de perto) não corresponde à câmera de poste do nosso sistema.
- **Seta não tem dataset de câmera fixa** → vamos treinar nosso próprio classificador de seta (pequeno, rotulado por nós).

## 3. Infrações no escopo

| Prioridade | Infração | Status | Como detecta |
| --- | --- | --- | --- |
| Alta | **Troca de faixa sem seta** | ✅ regra `lane_change_no_signal` (cena precisa de `lanes` + `lane_lines`); classificador de seta falta rotular/treinar | Máquina de estados de faixa + classificador de seta (FFT do brilho âmbar por lado) |
| Alta | **"Gato"** (ver definição abaixo) | ✅ `gato_canteiro`, `gato_re` | Ponto do veículo dentro de `forbidden_areas` / trajetória que inverte em linha reta (ré) |
| Alta | **Acidente / batida com fuga** ("bateu, saiu e não prestou socorro") | ⬜ não implementado | Ideia: IoU entre tracks + parada brusca = colisão; um dos tracks sai da cena logo após = fuga |
| Alta | **Leitura de placa** | ⬜ falta detector de placa BR + OCR (`perception/plate_reader.py` é stub) | YOLO 1 classe "plate" + EasyOCR + correção pela `PlateTrie` (Levenshtein) |
| Média | **Placa clonada** | ⬜ ideia | Grafo da cidade (câmeras = nós, tempo de trajeto = peso): mesma placa em 2 câmeras em tempo < Dijkstra = suspeita |
| — | Sinal vermelho, contramão, parada na faixa de pedestre, faixa de ônibus, linha contínua, conversão proibida | ✅ em `rules/violations/checks.py` | Geometria + grafo da via |
| Extra | Fila dupla, furar fila, celular, sem capacete | ⬜ | Ver doc de planejamento |

**"Gato" (definição do Lucas):** qualquer forma errada de continuar na via, cortando caminho por onde não deveria. Exemplos:
- pista de um lado, pista do outro, terra/canteiro no meio → o motorista **sobe na terra para contornar e fazer retorno forçado**;
- parado no sinaleiro, **dá ré**, entra um pouco na contramão e **sobe por outra rua**.
Na cena, marcar canteiros/ilhas/calçadas/terra como `forbidden_areas` (tecla **g** no `scripts/scene_editor.py`).

**CTB (conferir artigo/inciso antes de apresentar — vários foram citados de memória):**
sinal vermelho art. 208 · contramão art. 186 · conversão/retorno proibido arts. 206/207 · faixa de pedestre art. 183 · furar fila art. 211 · seta art. 196 · linha contínua arts. 185/203 · ônibus art. 184 · fila dupla art. 181 XI · gato no canteiro arts. 193/206 · marcha à ré art. 194 · omissão de socorro (batida com fuga) — verificar artigo.
Pontos: leve 3, média 4, grave 5, gravíssima 7. Multas: R$ 88,38 / 130,16 / 195,23 / 293,47.
Suspensão (art. 261, I, redação da Lei 14.071/2020 — conferido): 20 pts com 2+ gravíssimas, 30 com 1, 40 sem gravíssima. Motorista com EAR: sempre 40 (não modelado).
Pontos (art. 259), multas (art. 258) e advertência (art. 267) também conferidos.

## 4. Arquitetura

```
Câmera → [RingBuffer] → Detector YOLO → Tracker [Kalman + Húngaro] → Regras [geometria, grafo, FSM]
      → Evidência [SHA-256 encadeado] → Fila [MinHeap por gravidade] → Detran sim [Bloom + AVL + Fenwick] → Dashboard
```

| Pasta | Conteúdo |
| --- | --- |
| `eda/` | ring_buffer, geometry, hungarian, kalman, quadtree, road_graph (BFS/Dijkstra), priority_heap, plate_trie (+Levenshtein), bloom_filter, avl_tree (soma por intervalo), segment_tree (Fenwick) |
| `perception/` | detector (YOLO), tracker (SORT próprio), traffic_light (HSV), blinker (`BlinkerClassifier`: RandomForest por track, cai na heurística sem modelo), blinker_features (FFT por lado), plate_reader (stub) |
| `rules/` | engine, `violations/checks.py` (uma função por infração), state_machine (troca de faixa), types (pontos/multas) |
| `evidence/` | packager: clipe + frame + JSON + hash encadeado (cadeia de custódia) |
| `detran_sim/` | scoring (advertência, 3 leves, limite CNH) + API FastAPI |
| `dashboard/` | Streamlit |
| `benchmarks/` | nossa estrutura vs. ingênuo |
| `data/scenes/*.json` | geometria por câmera (stop_line, solid_lines, lane_lines, lanes, crosswalk, bus_lane, zones+direction, allowed_moves, forbidden_areas) |
| `scripts/` | prepare_visdrone, train (YOLO), run_pipeline, scene_editor, extract_vehicle_sequences, label_blinkers, train_blinker |

**Benchmarks medidos (para os slides):** quadtree 23x mais rápida que lista · AVL 508x mais rápida que varredura · heap 23x mais rápido que lista+sort.

## 5. Ambiente

- MacBook Air Apple Silicon. Python 3.9.6 em `.venv` (`source .venv/bin/activate`). torch 2.8.0, **MPS = True** (GPU do Mac ok). `bash setup.sh` já rodado com sucesso.
- `python -m pytest -q tests` → **18 testes** devem passar.
- No zsh **não colar comentários `# ...`** na mesma linha do comando (deu erro antes).
- Treino de seta precisa de `pip install scikit-learn joblib` (já no requirements.txt).
- **Treino YOLO no MPS: `amp=False`** (já no `train.py`) — com fp16 a loss vira NaN na 1ª época. Se o treino parar: `python scripts/train.py --resume runs/train/visdrone_vehicles/weights/last.pt`.
- **Problema conhecido:** a ponte do Claude do chat com o Mac cai muito (Mac dormindo / app fechado) e a VM Linux dela tem só ~10 GB (estourou ao instalar torch). Por isso o trabalho pesado passa a ser feito aqui, no Claude Code local. Para sessões longas: `caffeinate -dis` num terminal.

## 6. Dados

| Dataset | Onde | Uso | Status |
| --- | --- | --- | --- |
| VisDrone2019-DET | `data/processed/visdrone` (YOLO, 7 classes: pedestrian, bicycle, car, van, truck, bus, motor) | Fine-tune do detector | ✅ convertido (6471 train / 548 val / 1610 test). Visão aérea/alta |
| **WIBAM demo** (Adelaide, Austrália, CC BY 4.0) | `data/raw/wibam_demo/` (frames/0..3, annotations, calib) | **Base do treino de seta** | ✅ descompactado. 4 câmeras fixas num cruzamento, 1920x1080, ~400 frames/câmera em 2 trechos contínuos. Câmera baixa e próxima — lanterna visível |
| OpenALPR benchmarks | `data/raw/benchmarks-master/endtoend/br` (229 placas BR) | Testar OCR de placa | ✅ |
| COCO128 · YOLO11 n/s/m | `data/raw/coco128`, `models/pretrained/` | Smoke test / ponto de partida | ✅ |
| Vídeos de exemplo | `data/videos/` | Testar pipeline | ✅ mas são de **estacionamento**; cena `exemplo_car_detection.json` é fictícia |
| UC3M-LP (placas, Espanha, CC BY 4.0, 4,5 GB) | `~/Downloads/UC3M-LP.zip` | Detector de placa | ⚠️ download pode ter parado no meio — conferir tamanho (4547680070 bytes) |
| TUMTraf Intersection (Munique, CC BY-NC-ND) | — | Cruzamento, manobras (conversões, retorno, ultrapassagem) | ⬜ exige cadastro: https://a9-dataset.innovation-mobility.com/en/register |
| ACCIDENT (CCTV real + sintético CARLA, 5 tipos de colisão) | — | Acidentes | ⬜ exige login Kaggle (`picekl/accident`); repo github.com/accidentbench/ACCIDENT |
| NGSIM Lankershim/Peachtree (EUA, público) | — | Troca de faixa, semáforo, trajetórias | ⬜ data.transportation.gov |
| LUMPI (Hannover) | — | Cruzamento multi-câmera | ⬜ data.uni-hannover.de |
| RodoSol / UFPR-ALPR | — | Placas BR | ⬜ formulário dos autores |
| **Vídeo próprio de Goiânia** | — | Teste real + demo do "gato" | ⬜ gravar 10–20 min de uma passarela (retorno com canteiro de terra, cruzamentos tipo T-63 / Av. 85). Borrar placas/rostos no material público |

**Limpeza pendente:** em `data/raw/` dá para apagar `dl.log`, `dl.sh`, `_extract/` e os zips do VisDrone (~1,9 GB). Em `~/Downloads` sobrou uma cópia de `wibam_demo.zip` (1,1 GB).

## 7. Treino de seta (pipeline)

1. `python scripts/extract_vehicle_sequences.py --frames data/raw/wibam_demo/frames/2 --fps 50 --prefix wibam_c2` (repetir para câmeras 0, 1, 3). **fps 50 é estimado** pelo próprio pisca (WIBAM não informa fps): em `wibam_c1_c0_t6` a seta acende a cada ~34 frames; norma = 1–2 Hz ⇒ 34–68 fps; 50 fps ⇒ 1,47 Hz (típico). Com 25 o pisca caía em 0,7 Hz, fora da banda das features.
2. `python scripts/label_blinkers.py` — humano rotula: **n** nenhuma, **e** esquerda, **d** direita, **a** alerta, **x** descartar, **b** voltar, **q** sair. Lado = do ponto de vista do motorista.
3. `python scripts/train_blinker.py` → `models/blinker_rf.joblib` (RandomForest, features FFT, aumento por espelhamento, validação por sequência). Com dados sintéticos deu 100% — número real só com o WIBAM.
4. ✅ Já integrado: `run_pipeline.py` usa o modelo se `models/blinker_rf.joblib` existir; regra `lane_change_no_signal` lê `track.state["blinker"]`.

Achados na inspeção (mapa de energia 0,8–2,5 Hz por pixel): **pisca confirmado em `wibam_c1_c0_t6` e `wibam_c3_c1_t4`** (seta dianteira, ciclos regulares). `c1_c1_t16`, `c0_c0_t29` = luz de freio (acende e fica). Ou seja, **seta aparece no WIBAM, mas é rara (~2–5 de 72)** — poucos positivos para treinar; vídeo próprio continua importante.
Obs.: as 72 sequências já foram extraídas (`data/processed/blinker/seqs`). As de maior energia na banda 0,8–2,5 Hz eram oclusão, não pisca — rótulo manual é indispensável.
Risco: a seta pode não ser visível o bastante no WIBAM. Se não for, plano B é gravar vídeo próprio.

## 8. Onde paramos

- [x] Ideia estruturada + doc de planejamento (claude.ai: "CompQuest — Fiscalização de Trânsito por Câmeras")
- [x] Projeto, `eda/`, pipeline, regras, Detran sim, dashboard, benchmarks — testes passando
- [x] Ambiente no Mac com MPS
- [x] VisDrone convertido · WIBAM baixado · `train.py` corrigido (YAML com caminho absoluto) e testado
- [x] Regras do "gato" + scripts do treino de seta
- [x] `compquest_update.zip` aplicado + regra de troca de faixa sem seta + seta integrada ao pipeline — 18 testes
- [x] Sequências de seta extraídas do WIBAM (72)
- [ ] **EM ANDAMENTO:** treino YOLO VisDrone yolo11s 50 épocas (começou 13/11 ~10:50, ~12 h). Log: `runs/logs/train_visdrone.log`
- [x] Seta rotulada (72: 5 direita, 2 esquerda, 65 nenhuma) e treinada → **não funcionou: 0/7 setas achadas na validação cruzada.** Modelo guardado como `models/blinker_rf_v0_nao_usar.joblib` (fora do caminho padrão, para o pipeline não usar).
  Diagnóstico: 7 positivos é pouco; carros em movimento + recorte tremendo; maiores falsos positivos = ônibus (fundo na borda do recorte) e luz de freio. Testado: features por pixel, janelas menores, supervisão fraca, autocorrelação — melhora o ranking mas não separa.
  **Enquanto a seta não funcionar, NÃO ativar `lanes` nas cenas**: a regra de troca de faixa acusaria todo mundo de "sem seta".
- [ ] **PRÓXIMO (seta):** mais positivos e mais fáceis — vídeo próprio com carros **parados no sinal esperando para virar** (seta piscando com o carro parado é o caso limpo), vista traseira. Meta: 30+ sequências com seta.
- [ ] Rodar `run_pipeline.py` com o `best.pt` do VisDrone e montar cena WIBAM (`scene_editor.py --image`)
- [ ] Regra de acidente / batida com fuga
- [ ] Detector de placa (UC3M-LP / RodoSol) + OCR + regra de placa clonada
- [ ] Cadastro TUMTraf e login Kaggle (Lucas)
- [ ] Gravar vídeos em Goiânia e montar cenas com o `scene_editor.py`
- [ ] Conferir artigos do CTB restantes (linha contínua: 185 média vs. 203 V gravíssima; gato; omissão de socorro)
- [ ] Slides: pipeline, tabela de EDA com complexidades, benchmarks, demo em vídeo

Histórico de sessões: `docs/SESSAO_2026-11-13.md` (feito, problemas, resultado da seta, próximos passos).

## 9. Ideias extras (diferenciais)

- **Processamento na borda:** só o clipe da infração sai da câmera (ring buffer + heap justificam com números de banda).
- **Cadeia de custódia:** hash SHA-256 encadeado de cada evidência (já implementado).
- **Revisão humana:** evento de baixa confiança vai para fila de revisão, não vira multa automática.
- **Mapa de calor educativo:** ranking de cruzamentos/horários com mais infrações (Fenwick) para orientar campanhas.
- **Placa clonada via Dijkstra** no grafo de câmeras.
- **Limitações a apresentar com honestidade:** domain gap (dados de outros países), seta é a mais difícil, cada câmera precisa de calibração (JSON de cena), autuação real exige equipamento homologado e agente — o sistema é de apoio; LGPD (borrar placas/rostos).
