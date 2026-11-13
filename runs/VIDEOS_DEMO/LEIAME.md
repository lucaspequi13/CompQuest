# Vídeos da demo — CompQuest

Todos os vídeos anotados ficam aqui. Os dados de cada rodada (infracoes.jsonl, revisao.jsonl, evidence/)
continuam em `runs/demo/<nome>/`.

Legenda no vídeo: 🟢 virou com seta · 🔴 infração · 🟡 incerto (revisão humana) · círculo vermelho = BATIDA · rastro vermelho = FUGA / infração.
"(roteiro)" e a faixa "MODO ROTEIRO" no rodapé = algo anotado à mão; o resto (detecção, tracking, semáforo, regras) é do sistema.

**Depois de toda batida o sistema acompanha por 4 s se os envolvidos param** (texto "ACOMPANHANDO SE PARAM"):
- **FICARAM NO LOCAL** (verde) → pararam / prestaram socorro;
- **FUGA** (rastro vermelho) → um foi embora e o outro ficou → `evasao_apos_colisao` (CTB art. 176 I / 305 — conferir);
- **VIDEO ACABA ANTES: SOCORRO NAO OBSERVADO** → o clipe termina antes dos 4 s → `colisao_inconclusiva` na revisão humana.
Os clipes dos datasets (TU-DAT, ACCIDENT) são **cortados pelos autores logo depois da batida**; não existe continuação nos arquivos.

**Placas:** canto superior direito, "PLACAS MONITORADAS: car #8 - placa: ILEGIVEL". O sistema sempre registra o
melhor recorte da placa de cada veículo (detector de placa yolo-v9 do open-image-models + OCR fast-plate-ocr,
qualquer país) e manda placa + recorte (`placa.jpg`, dentro do hash da evidência) para o Detran.
Nesses vídeos dá ILEGIVEL: baixa resolução / noite / longe, e alguns datasets **borram as placas** (anonimização).

## 1_finais (modelo final `models/detector_final.pt`)
| Arquivo | Origem | O que mostra | Anotado à mão |
| --- | --- | --- | --- |
| 01_avenida_vista_de_cima_entrou_sem_seta.mp4 | Pexels 33585547 (drone parado) | Carro sai da rua lateral e entra na avenida sem seta (~40 s) | a seta |
| 02_cruzamento_batida_fuga_e_seta.mp4 | UA-DETRAC MVI_40701 | Batida encenada → fuga; seta OK; sem seta; conversões amarelas. Bagunçado: a via permite quase tudo | seta e batida |
| **03_avancou_vermelho_e_bateu.mp4** | TU-DAT v36 (câmera de rua, EUA, noite) | Semáforo lido sozinho (AMARELO → VERMELHO) → carro cruza a linha no vermelho → bate no carro branco → vídeo acaba antes de dar para ver se pararam | só o momento da batida |
| **04_avancou_vermelho_e_bateu_camera_fiscalizacao.mp4** | TU-DAT v40 (câmera de fiscalização "American Traffic Solutions", via Al Jazeera) | SUV avança o vermelho → bate no carro branco → vídeo acaba antes | só o momento da batida |
| **05_tunel_bateu_e_fugiu.mp4** | ACCIDENT (prévia pública, "sideswipe") | Caminhão raspa num carro, o carro capota e **o caminhão segue sem parar** até onde a câmera vê | batida e fuga (carro capotado some da detecção; fuga em profundidade quase não anda em pixels) |

| **06_EPTV_av_saudade_derrubou_moto_e_fugiu.mp4** | Reportagem do **Jornal da EPTV** (Av. da Saudade), gravada pelo Lucas (`data/gravacao/`). Recortada só a imagem da câmera (`prepare_video.py --crop 0 180 480 545 --scale 2 --end 6.5`, tira logo/legenda/replay) | Carro troca de pista **sem seta** (rota vermelha "SEM SETA troca faixa", desenhada sobre a trajetória real), derruba a motociclista e **vai embora**: BATIDA → "ACOMPANHANDO SE PARAM" → **FUGA #6 decidida pelo sistema**. Placa monitorada desde a manobra. 2 infrações: troca_faixa_sem_seta + evasao_apos_colisao | seta = "nenhuma" (dito na reportagem), QUANDO foi a troca de pista (a perspectiva esconde a troca de faixa: ~8° e ~15 px), momento da batida e ponto onde a moto caiu |

Créditos nos slides: EPTV (06); TU-DAT (Kumar & Kant, *Sensors* 2025, uso só para pesquisa) + fontes que aparecem no vídeo (03/04);
ACCIDENT (Picek et al., CVPRW 2026) (05).

## 2_versoes_antigas
Mesmos vídeos 01/02 com modelos anteriores (VisDrone / fine-tune parcial) ou sem roteiro. Só para comparar.

## imagens
Prévias (frames lado a lado) dos vídeos acima.
