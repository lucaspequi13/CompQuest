"""Leitura de placa: detectar a placa no recorte do veículo + OCR + correção pela trie.

Status: FALTA um detector de placa BR treinado. Caminho sugerido:
  1. Pedir acesso ao RodoSol-ALPR / UFPR-ALPR (formulário dos autores)
  2. Treinar YOLO de 1 classe "plate" com scripts/train.py --data configs/plates.yaml
  3. OCR com EasyOCR ou PaddleOCR no recorte
Enquanto isso, read_plate() devolve None e o pipeline usa o track ID.
"""
from typing import Optional

from eda.plate_trie import is_valid_plate, normalize


class PlateReader:
    def __init__(self, plate_weights: Optional[str] = None):
        self.detector = None
        self.ocr = None
        if plate_weights:
            from ultralytics import YOLO
            self.detector = YOLO(plate_weights)
            try:
                import easyocr
                self.ocr = easyocr.Reader(["en"], gpu=False)
            except ImportError:
                self.ocr = None

    def read_plate(self, vehicle_crop) -> Optional[str]:
        if self.detector is None or self.ocr is None:
            return None
        res = self.detector.predict(vehicle_crop, verbose=False)[0]
        if not len(res.boxes):
            return None
        x1, y1, x2, y2 = map(int, res.boxes.xyxy[0].tolist())
        texts = self.ocr.readtext(vehicle_crop[y1:y2, x1:x2], detail=0)
        cand = normalize("".join(texts))
        return cand if is_valid_plate(cand) else None
