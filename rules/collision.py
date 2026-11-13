"""Batida e batida com fuga, a partir dos tracks (sem rede neural).

Colisão (candidata): duas caixas se tocam (IoU > iou_min), os "pés" das caixas estão perto no chão
  e pelo menos um dos dois PAROU DE REPENTE: vinha andando (mediana da velocidade no início da janela)
  e está parado há pelo menos stop_s. Frear normal leva segundos; batida, frações.
Fuga: depois da colisão, dentro de watch_s, um envolvido vai embora (se afasta do ponto da batida
  andando, ou some da cena andando) enquanto o outro continua parado no local. Quem foi = evasão.
  Se os dois ficam: só alerta de colisão. Se os dois saem: revisão humana.

Filtros contra falso positivo em vídeo real: caixas pequenas (longe, no fundo da perspectiva) e
tracks muito novos são ignorados; cada veículo é acusado de fuga no máximo uma vez.

EDA: os pares candidatos vêm da QuadTree (busca por raio) em vez de testar todos os pares.
Velocidade em "alturas de caixa por segundo", para não depender da distância até a câmera.

CTB (CONFERIR antes de apresentar): omissão de socorro art. 176, I (gravíssima);
afastar-se do local para fugir à responsabilidade = crime do art. 305.
"""
import math
from collections import deque
from statistics import median

from eda.geometry import iou
from eda.quadtree import QuadTree, Rect


def _speed(track, fps, k):
    tr = track.trail
    if len(tr) <= k:
        return 0.0
    (xa, ya), (xb, yb) = tr[-1 - k], tr[-1]
    h = max(1.0, track.box[3] - track.box[1])
    return math.hypot(xb - xa, yb - ya) / h * fps / k


class CollisionWatcher:
    def __init__(self, fps: float, frame_size, iou_min: float = 0.05, move: float = 0.8, stop: float = 0.2,
                 window_s: float = 1.5, stop_s: float = 0.4, watch_s: float = 4.0, leave_widths: float = 2.5,
                 min_box_frac: float = 0.06, min_age_s: float = 0.5, k: int = 5):
        self.fps, self.k = fps, k
        self.w, self.h = frame_size
        self.iou_min, self.move, self.stop = iou_min, move, stop
        self.window = max(4, int(window_s * fps))
        self.stop_n = max(2, int(stop_s * fps))
        self.watch = int(watch_s * fps)
        self.leave_widths = leave_widths
        self.min_box_h = min_box_frac * self.h
        self.min_age = int(min_age_s * fps)
        self.speeds = {}      # track_id -> deque de velocidades recentes
        self.last = {}        # track_id -> último estado conhecido (para quem some)
        self.open = []        # colisões em observação
        self.seen_pairs = set()
        self.accused = set()

    def _abrupt_stop(self, tid):
        s = self.speeds.get(tid)
        if not s or len(s) < self.stop_n * 2:
            return False
        s = list(s)
        before, after = s[:-self.stop_n], s[-self.stop_n:]
        return median(before[: max(1, len(before) // 2)]) >= self.move and max(after) <= self.stop

    def _eligible(self, t):
        x1, y1, x2, y2 = t.box
        m = 0.01 * max(self.w, self.h)  # caixa cortada na borda: o "pé" para de andar e parece batida
        inside = x1 > m and y1 > m and x2 < self.w - m and y2 < self.h - m
        return inside and (y2 - y1) >= self.min_box_h and t.hits >= self.min_age

    def force(self, id_a: int, id_b: int, point, frame_idx: int, leaver=None):
        """Batida marcada no roteiro. Sem `leaver`, o sistema decide quem fugiu (lógica normal).
        Com `leaver` (roteiro diz quem foi embora), a fuga sai 1,5 s depois, marcada como roteiro."""
        self.seen_pairs.add((min(id_a, id_b), max(id_a, id_b)))
        col = {"tipo": "colisao", "ids": [id_a, id_b], "frame": frame_idx, "ponto": tuple(point), "roteiro": True,
               "leaver": leaver}
        self.open.append(col)
        return dict(col)

    def update(self, tracks, alive_ids, frame_idx: int):
        """tracks: confirmados neste frame. alive_ids: ids que o tracker ainda mantém (inclui perdidos há pouco)."""
        events = []
        for t in tracks:
            v = _speed(t, self.fps, self.k)
            self.speeds.setdefault(t.id, deque(maxlen=self.window)).append(v)
            x1, y1, x2, y2 = t.box
            self.last[t.id] = {"point": t.point, "speed": v, "width": x2 - x1, "frame": frame_idx,
                               "trail": list(t.trail), "cls": t.cls}

        # 1) colisões novas: pares próximos via QuadTree
        cand = [t for t in tracks if self._eligible(t)]
        qt = QuadTree(Rect(-self.w, -self.h, 3 * self.w, 3 * self.h))
        for t in cand:
            qt.insert(*t.point, t)
        for a in cand:
            size_a = max(a.box[2] - a.box[0], a.box[3] - a.box[1])
            for _, _, b in qt.query_radius(*a.point, 1.5 * size_a):
                if b.id <= a.id or (a.id, b.id) in self.seen_pairs:
                    continue
                if iou(a.box, b.box) < self.iou_min:
                    continue
                ground = math.hypot(a.point[0] - b.point[0], a.point[1] - b.point[1])
                if ground > 0.8 * max(a.box[3] - a.box[1], b.box[3] - b.box[1]):
                    continue  # um atrás do outro na perspectiva, não encostados no chão
                if not (self._abrupt_stop(a.id) or self._abrupt_stop(b.id)):
                    continue
                self.seen_pairs.add((a.id, b.id))
                pt = ((a.point[0] + b.point[0]) / 2, (a.point[1] + b.point[1]) / 2)
                col = {"tipo": "colisao", "ids": [a.id, b.id], "frame": frame_idx, "ponto": pt}
                self.open.append(col)
                events.append(dict(col))

        # 2) decidir as colisões em observação
        still = []
        for col in self.open:
            lv = col.get("leaver")
            if lv is not None:  # fuga dada pelo roteiro
                if frame_idx - col["frame"] >= int(1.5 * self.fps) or lv not in alive_ids:
                    if lv not in self.accused and lv in self.last:
                        self.accused.add(lv)
                        events.append({"tipo": "fuga", "track_id": lv, "ficou": [i for i in col["ids"] if i != lv],
                                       "frame": frame_idx, "ponto": col["ponto"], "trilha": self.last[lv]["trail"],
                                       "cls": self.last[lv]["cls"], "roteiro": True, "fuga_do_roteiro": True})
                    continue
                still.append(col)
                continue
            gone_ids, stayed = [], []
            for tid in col["ids"]:
                info = self.last.get(tid)
                if info is None:
                    continue
                far = math.hypot(info["point"][0] - col["ponto"][0], info["point"][1] - col["ponto"][1])
                left_area = far > self.leave_widths * max(info["width"], 1.0) and info["speed"] >= self.move
                vanished = tid not in alive_ids and info["speed"] >= self.move
                if left_area or vanished:
                    gone_ids.append(tid)
                elif tid in alive_ids and info["speed"] <= 2 * self.stop:
                    stayed.append(tid)
            if gone_ids and stayed:
                for tid in gone_ids:
                    if tid in self.accused:
                        continue
                    self.accused.add(tid)
                    events.append({"tipo": "fuga", "track_id": tid, "ficou": stayed, "frame": frame_idx,
                                   "ponto": col["ponto"], "trilha": self.last[tid]["trail"],
                                   "cls": self.last[tid]["cls"], "roteiro": col.get("roteiro", False)})
                continue
            if frame_idx - col["frame"] >= self.watch:
                if len(gone_ids) == len(col["ids"]):
                    events.append({"tipo": "colisao_revisao", "ids": col["ids"], "frame": frame_idx,
                                   "ponto": col["ponto"]})
                continue
            still.append(col)
        self.open = still

        # 3) limpeza de memória de quem já sumiu e não está numa colisão aberta
        keep = set(alive_ids) | {tid for c in self.open for tid in c["ids"]}
        for d in (self.speeds, self.last):
            for tid in [t for t in d if t not in keep]:
                del d[tid]
        return events
