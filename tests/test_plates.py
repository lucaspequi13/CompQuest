"""Registro de placa: guarda o recorte mais nítido por veículo, marca ILEGIVEL sem OCR, vai para a evidência."""
import os

import numpy as np

from evidence.packager import EvidencePackager
from perception.detector import Detection
from perception.plate_reader import ILEGIVEL, PlateRegistry, plate_region, sharpness
from perception.tracker import Tracker
from rules.types import Violation


def _frame(sharp: bool):
    f = np.full((240, 320, 3), 90, np.uint8)
    # "placa" na parte de baixo do carro (caixa 100..220 x 100..200): listras = nítido, borrão = liso
    if sharp:
        f[170:190, 140:180:4] = 255
    else:
        f[170:190, 140:180] = 170
    return f


def test_keeps_sharpest_crop_and_marks_illegible_without_ocr():
    reg, tr = PlateRegistry(), Tracker(min_hits=1)
    box = (100, 100, 220, 200)
    for sharp in (False, True, False):
        tracks = tr.update([Detection(box, "car", 0.9)])
        reg.update(_frame(sharp), tracks)
    tid = tracks[0].id
    crop = reg.best(tid)
    assert crop is not None and sharpness(crop) > 100       # ficou com o quadro nítido
    assert reg.text(tid) == ILEGIVEL                        # sem detector/OCR: não inventa texto
    x1, y1, x2, y2 = plate_region(box, (240, 320))
    assert 100 < x1 < x2 < 220 and 150 < y1 < y2 <= 200     # região: centro, parte de baixo
    reg.forget(set(), keep={tid})
    assert reg.best(tid) is not None
    reg.forget(set())
    assert reg.best(tid) is None


def test_plate_crop_goes_into_evidence(tmp_path):
    pack = EvidencePackager(str(tmp_path), fps=30)
    v = Violation("avanco_sinal_vermelho", "gravissima", "Art. 208", 7, 10, 0.0, "cam", (1, 2), plate=ILEGIVEL)
    folder = pack.package(v, [], _frame(True), plate_crop=_frame(True)[170:190, 140:180])
    assert os.path.exists(os.path.join(folder, "placa.jpg"))
    import json
    meta = json.load(open(os.path.join(folder, "evidence.json"), encoding="utf-8"))
    assert "placa.jpg" in meta["files"] and meta["plate"] == ILEGIVEL and len(meta["hash"]) == 64
