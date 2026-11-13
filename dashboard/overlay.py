"""Desenho da demo: rastro colorido da manobra com seta de direção, marca de batida e fuga, legenda.

Cada desenho fica na tela por `hold_s` segundos depois do evento, mesmo que o veículo saia de cena.
"""
import cv2
import numpy as np

BGR = {"verde": (60, 200, 60), "vermelho": (40, 40, 235), "amarelo": (0, 215, 255),
       "cinza": (170, 170, 170), "branco": (255, 255, 255)}
LABEL = {"verde": "SETA OK", "vermelho": "SEM SETA", "amarelo": "SETA ?"}


def put_text(img, text, org, color, scale=0.55, thick=1):
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_path(img, pts, color, thick=3, arrow=True):
    pts = [(int(x), int(y)) for x, y in pts]
    if len(pts) < 2:
        return
    cv2.polylines(img, [np.array(pts, np.int32)], False, color, thick, cv2.LINE_AA)
    if arrow:
        tail = pts[max(0, len(pts) - 8)]
        if tail != pts[-1]:
            cv2.arrowedLine(img, tail, pts[-1], color, thick + 1, cv2.LINE_AA, tipLength=0.6)


class Overlay:
    def __init__(self, fps: float, hold_s: float = 3.0):
        self.hold = int(hold_s * fps)
        self.items = []  # (até_frame, tipo, dados)
        self.roteiro = False  # modo roteiro: mostra no vídeo que seta/batida foram anotadas à mão

    def add_maneuver(self, ev):
        self.items.append((ev["frame"] + self.hold, "manobra", ev))

    def add_collision(self, ev):
        self.items.append((ev["frame"] + self.hold * 2, "colisao", ev))

    def add_fuga(self, ev):
        self.items.append((ev["frame"] + self.hold * 2, "fuga", ev))

    def draw(self, img, frame_idx, tracks):
        # rastro fino de todos os veículos (contexto)
        for t in tracks:
            draw_path(img, list(t.trail)[-30:], BGR["cinza"], 1, arrow=False)
        self.items = [it for it in self.items if it[0] >= frame_idx]
        live = {t.id: t for t in tracks}
        for _, kind, ev in self.items:
            if kind == "manobra":
                pts = list(ev["pontos"])
                t = live.get(ev["track_id"])
                if t is not None:  # continua o rastro enquanto o carro anda
                    pts += [p for p in list(t.trail)[-15:] if p not in pts]
                c = BGR[ev["cor"]]
                draw_path(img, pts, c, 3)
                end = pts[-1]
                tipo = "conversao" if ev["tipo"] == "conversao" else "troca faixa"
                put_text(img, f"{LABEL[ev['cor']]} {tipo} {ev['lado']}", (end[0] + 8, end[1] - 8), c, 0.55, 2)
                if ev.get("roteiro"):
                    put_text(img, "(seta: roteiro)", (end[0] + 8, end[1] + 10), BGR["branco"], 0.42, 1)
            elif kind == "colisao":
                x, y = map(int, ev["ponto"])
                r = 18 + 6 * (frame_idx % 10 < 5)  # pisca
                cv2.circle(img, (x, y), r, BGR["vermelho"], 3, cv2.LINE_AA)
                put_text(img, "BATIDA" + (" (roteiro)" if ev.get("roteiro") else ""), (x - 30, y - r - 8),
                         BGR["vermelho"], 0.7, 2)
            elif kind == "fuga":
                t = live.get(ev["track_id"])
                pts = list(ev["trilha"]) + (list(t.trail)[-30:] if t is not None else [])
                draw_path(img, pts, BGR["vermelho"], 4)
                end = pts[-1]
                put_text(img, f"FUGA #{ev['track_id']}", (end[0] + 8, end[1] + 18), BGR["vermelho"], 0.7, 2)
        self._legend(img)
        if self.roteiro:
            h = img.shape[0]
            put_text(img, "MODO ROTEIRO: seta e momento da batida anotados a mao; deteccao, tracking e regras reais",
                     (10, h - 12), BGR["amarelo"], 0.45, 1)

    @staticmethod
    def _legend(img):
        h, w = img.shape[:2]
        rows = [("verde", "virou com seta"), ("vermelho", "infracao"), ("amarelo", "incerto: revisao humana")]
        x0, y0 = w - 250, 12
        cv2.rectangle(img, (x0 - 8, y0 - 4), (w - 6, y0 + 22 * len(rows) + 2), (0, 0, 0), -1)
        for i, (c, txt) in enumerate(rows):
            y = y0 + 22 * i + 14
            cv2.line(img, (x0, y - 4), (x0 + 26, y - 4), BGR[c], 4)
            put_text(img, txt, (x0 + 34, y), BGR["branco"], 0.45, 1)
