"""Wrapper do YOLO (Ultralytics). Única parte "caixa-preta" do pipeline.

Retorna detecções como dicts simples para o resto do código não depender da lib.
"""
from dataclasses import dataclass
from typing import List

# Classes COCO que interessam (modelo pré-treinado)
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle", 0: "person", 9: "traffic_light"}
# O que a fiscalização olha (nomes do nosso modelo e do COCO). Pedestre/bicicleta ficam de fora.
VEHICLES = {"car", "van", "truck", "bus", "motor", "motorcycle"}


@dataclass
class Detection:
    box: tuple  # (x1, y1, x2, y2)
    cls: str
    conf: float


class Detector:
    def __init__(self, weights: str = "models/pretrained/yolo11n.pt", conf: float = 0.35, device: str = None,
                 keep=None):
        """keep: nomes de classe que passam (ex.: VEHICLES). None = todas."""
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.keep = set(keep) if keep else None
        self.device = device  # None = auto ("mps" no Mac M1/M2/M3)
        names = self.model.names
        self.is_coco = len(names) == 80

    def __call__(self, frame) -> List[Detection]:
        # agnostic_nms: o mesmo carro às vezes vem como "car" E "van"; sem isso viram 2 tracks e a trajetória racha
        res = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False, agnostic_nms=True)[0]
        out: List[Detection] = []
        for b in res.boxes:
            cid = int(b.cls)
            if self.is_coco:
                if cid not in COCO_VEHICLES:
                    continue
                name = COCO_VEHICLES[cid]
            else:
                name = self.model.names[cid]
            if self.keep is not None and name not in self.keep:
                continue
            out.append(Detection(tuple(float(v) for v in b.xyxy[0].tolist()), name, float(b.conf)))
        return out
