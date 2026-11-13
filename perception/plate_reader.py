"""Registro de placa por veículo: sempre guarda o MELHOR recorte da placa e tenta ler o texto.

Câmera de rua/fiscalização nem sempre deixa ler a placa (longe, noite, borrão). Mesmo assim o sistema
registra, para cada veículo, o recorte mais nítido da região da placa visto até agora — é o que vai
para a evidência e para o Detran ("ILEGIVEL" + imagem), e um agente humano confere.

Camadas:
  1. Região da placa: com detector de placa (YOLO 1 classe, opcional) usa a caixa dele; sem detector,
     usa a região onde a placa costuma ficar (centro da parte de baixo da caixa do veículo).
  2. Melhor recorte por veículo: nota = nitidez (variância do Laplaciano) x área. Guarda o máximo.
  3. OCR (opcional, fast-plate-ocr ou EasyOCR) no melhor recorte; texto normalizado e conferido no
     formato BR (antigo AAA9999 e Mercosul AAA9A99) pela PlateTrie / Levenshtein do `eda/`.
"""
from typing import Dict, Optional

import cv2
import numpy as np

from eda.plate_trie import is_valid_plate, normalize

ILEGIVEL = "ILEGIVEL"


def plate_region(box, frame_shape):
    """Região provável da placa dentro da caixa do veículo (centro, parte de baixo)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rx1, rx2 = x1 + 0.22 * w, x2 - 0.22 * w
    ry1, ry2 = y1 + 0.58 * h, y2 - 0.04 * h
    H, W = frame_shape[:2]
    return (int(max(0, rx1)), int(max(0, ry1)), int(min(W, rx2)), int(min(H, ry2)))


def sharpness(img) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


class PlateRegistry:
    """Um por câmera. update() por frame; best()/text() por track."""

    def __init__(self, plate_weights: Optional[str] = None, ocr: Optional[str] = None, min_side: int = 12,
                 detect_every: int = 5):
        """plate_weights: caminho de um YOLO 1 classe 'plate', ou "oim" (detector pronto do
        open-image-models, yolo-v9 de placa). ocr: "fast" (fast-plate-ocr) ou "easy" (easyocr)."""
        self.detector, self.det_kind = None, None
        if plate_weights == "oim":
            from open_image_models import create_detector
            self.detector = create_detector("yolo-v9-t-640-license-plate-end2end", conf_thresh=0.15)
            self.det_kind = "oim"
        elif plate_weights:
            from ultralytics import YOLO
            self.detector = YOLO(plate_weights)
            self.det_kind = "yolo"
        self.detect_every = detect_every
        self.found: Dict[int, bool] = {}  # o detector achou placa de verdade neste veículo?
        self.loc: Dict[int, tuple] = {}   # onde a placa está AGORA (detectada ou região provável)
        self.hit_rel: Dict[int, tuple] = {}  # última placa detectada, relativa à caixa (acompanha o carro)
        self.ocr_name, self.ocr = None, None
        if ocr == "fast":
            from fast_plate_ocr import LicensePlateRecognizer
            self.ocr = LicensePlateRecognizer("cct-xs-v2-global-model", device="cpu")
            self.ocr_name = "fast"
        elif ocr == "easy":
            import easyocr
            self.ocr = easyocr.Reader(["en"], gpu=True)
            self.ocr_name = "easy"
        self.min_side = min_side
        self.crops: Dict[int, np.ndarray] = {}
        self.scores: Dict[int, float] = {}
        self.texts: Dict[int, str] = {}
        self.read_at: Dict[int, float] = {}

    # ---- captura -------------------------------------------------------------------------
    def _detect(self, frame, box):
        """Placa achada pelo detector dentro do recorte AMPLIADO do veículo, ou None."""
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if self.detector is None or crop.size == 0:
            return None
        s = max(1.0, 640 / max(crop.shape[:2]))
        big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else crop
        if self.det_kind == "oim":
            res = self.detector.predict(big)
            if not res:
                return None
            bb = max(res, key=lambda r: r.confidence).bounding_box
            px1, py1, px2, py2 = bb.x1, bb.y1, bb.x2, bb.y2
        else:
            r = self.detector.predict(big, conf=0.25, verbose=False)[0]
            if not len(r.boxes):
                return None
            px1, py1, px2, py2 = r.boxes.xyxy[0].tolist()
        return (int(x1 + px1 / s), int(y1 + py1 / s), int(x1 + px2 / s), int(y1 + py2 / s))

    def update(self, frame, tracks, frame_idx: int = 0, ignore=()) -> None:
        """ignore: retângulos [x1, y1, x2, y2] com sobreposições gravadas no vídeo (logo, botão do player,
        marca d'água). Placa detectada ali dentro é descartada — o detector confunde texto de UI com placa."""
        for t in tracks:
            hit = None
            big_enough = (t.box[3] - t.box[1]) >= 40  # carro de 20 px nunca tem placa legível: não gasta detector
            if self.detector is not None and big_enough and (frame_idx + t.id) % self.detect_every == 0:
                hit = self._detect(frame, t.box)
                if hit is not None:
                    cx, cy = (hit[0] + hit[2]) / 2, (hit[1] + hit[3]) / 2
                    if any(a <= cx <= c and b <= cy <= d for a, b, c, d in ignore):
                        hit = None
            # localização atual da placa (para o quadradinho): detecção deste frame, senão a última detecção
            # reposicionada na caixa atual, senão a região provável
            bx1, by1, bx2, by2 = t.box
            bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
            if hit is not None:
                self.hit_rel[t.id] = ((hit[0] - bx1) / bw, (hit[1] - by1) / bh, (hit[2] - bx1) / bw, (hit[3] - by1) / bh)
            rel = self.hit_rel.get(t.id)
            self.loc[t.id] = (bx1 + rel[0] * bw, by1 + rel[1] * bh, bx1 + rel[2] * bw, by1 + rel[3] * bh) if rel \
                else plate_region(t.box, frame.shape)
            if hit is None and self.found.get(t.id):
                continue  # já tem placa detectada de verdade: não troca por região genérica
            rx1, ry1, rx2, ry2 = hit or plate_region(t.box, frame.shape)
            # região tocando uma sobreposição gravada (botão, logo): não serve nem como recorte de reserva
            if any(rx1 < c and a < rx2 and ry1 < d and b < ry2 for a, b, c, d in ignore):
                continue
            if rx2 - rx1 < self.min_side or ry2 - ry1 < self.min_side // 2:
                continue
            crop = frame[ry1:ry2, rx1:rx2]
            score = sharpness(crop) * (rx2 - rx1) * (ry2 - ry1) * (10.0 if hit else 1.0)
            if hit and not self.found.get(t.id):
                self.scores[t.id] = -1.0  # primeira detecção real sempre substitui a região genérica
            if score > self.scores.get(t.id, -1.0):
                self.scores[t.id] = score
                self.crops[t.id] = crop.copy()
                if hit:
                    self.found[t.id] = True

    # ---- leitura -------------------------------------------------------------------------
    def _ocr(self, crop) -> Optional[str]:
        if self.ocr is None:
            return None
        try:
            if self.ocr_name == "fast":
                out = self.ocr.run(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))  # modelo v2: imagem colorida
                first = out[0] if isinstance(out, (list, tuple)) and out else out
                raw = getattr(first, "plate", first)
            else:
                raw = "".join(self.ocr.readtext(crop, detail=0))
        except Exception as e:  # OCR nunca derruba o pipeline: fica ILEGIVEL
            print("OCR falhou:", type(e).__name__, str(e)[:80])
            return None
        cand = normalize(str(raw).replace("_", ""))
        return cand or None

    def text(self, tid: int) -> str:
        """Texto lido da placa (qualquer país) ou ILEGIVEL.
        Os vídeos são de vários países: não exige o padrão BR. normalize() só corrige O/0, I/1... quando o
        texto tem 7 caracteres (formato BR); `is_br(texto)` diz se é placa brasileira."""
        crop = self.crops.get(tid)
        if crop is None:
            return ILEGIVEL
        if self.read_at.get(tid) != self.scores.get(tid):  # só relê se o recorte melhorou
            self.read_at[tid] = self.scores.get(tid)
            # OCR só em placa achada pelo detector (na região genérica ele inventa texto) ou sem detector
            cand = self._ocr(crop) if (self.found.get(tid) or self.detector is None) else None
            self.texts[tid] = cand if cand and len(cand) >= 4 else ILEGIVEL
        return self.texts.get(tid, ILEGIVEL)

    @staticmethod
    def is_br(txt: str) -> bool:
        return is_valid_plate(txt)

    def best(self, tid: int) -> Optional[np.ndarray]:
        return self.crops.get(tid)

    def forget(self, alive_ids, keep=()) -> None:
        keep = set(alive_ids) | set(keep)
        for d in (self.crops, self.scores, self.texts, self.read_at, self.found, self.loc, self.hit_rel):
            for k in [k for k in d if k not in keep]:
                del d[k]


# compatibilidade com o código antigo
class PlateReader(PlateRegistry):
    def read_plate(self, vehicle_crop) -> Optional[str]:
        txt = self._ocr(vehicle_crop)
        return txt if txt and is_valid_plate(txt) else None
