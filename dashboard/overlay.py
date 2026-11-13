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
        self.active = []

    def add_maneuver(self, ev):
        self.items.append((ev["frame"] + self.hold, "manobra", ev))

    def add_collision(self, ev):
        self.items.append((ev["frame"] + self.hold * 2, "colisao", ev))

    def add_fuga(self, ev):
        self.items.append((ev["frame"] + self.hold * 2, "fuga", ev))

    def track_color(self, tid, frame_idx):
        """Cor da caixa do veículo = cor do evento dele que está na tela (para saber de quem é cada rota).
        Prioridade: vermelho (infração/batida/fuga) > amarelo > verde. None = sem evento (caixa branca)."""
        best = None
        rank = {"vermelho": 3, "amarelo": 2, "verde": 1}
        for until, kind, ev in self.items:
            if until < frame_idx:
                continue
            if kind == "manobra" and ev["track_id"] == tid:
                cor = ev["cor"]
            elif kind in ("alerta", "fuga") and ev["track_id"] == tid:
                cor = "vermelho"
            elif kind == "colisao" and tid in ev.get("ids", []):
                cor = "vermelho"
            elif kind == "ficaram" and tid in ev.get("ids", []):
                cor = "verde"
            else:
                continue
            if best is None or rank[cor] > rank[best]:
                best = cor
        return BGR[best] if best else None

    @staticmethod
    def draw_plate_box(img, loc, color, thick=1):
        """Quadradinho onde está a placa (mesmo ilegível/borrada: mostra que o sistema sabe onde ela está)."""
        if loc is None:
            return
        x1, y1, x2, y2 = map(int, loc)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

    def add_ficaram(self, ev):
        self.items.append((ev["frame"] + self.hold * 2, "ficaram", ev))

    @staticmethod
    def draw_observing(img, observing, frames_left_in_video, fps):
        """Embaixo de cada batida em observação: contagem regressiva, ou aviso de que o vídeo acaba antes."""
        for (x, y), secs in observing:
            if secs * fps > frames_left_in_video:
                txt, cor = "VIDEO ACABA ANTES: SOCORRO NAO OBSERVADO", BGR["branco"]
            else:
                txt, cor = f"ACOMPANHANDO SE PARAM: {secs:.1f}s", BGR["amarelo"]
            put_text(img, txt, (int(x) - 120, int(y) + 48), cor, 0.6, 2)

    @staticmethod
    def draw_plate_tags(img, tracks, has_plate):
        """Etiqueta pequena embaixo de cada veículo cuja placa já está sendo registrada."""
        for t in tracks:
            if has_plate(t.id):
                x1, _, _, y2 = map(int, t.box)
                put_text(img, "placa registrada", (x1, y2 + 12), BGR["cinza"], 0.35, 1)

    @staticmethod
    def draw_plate_panel(img, entries):
        """Lista curta no canto superior direito, embaixo da legenda: 'car #8 · placa: ILEGIVEL'.
        Só para mostrar que a placa de cada infrator está sendo captada e monitorada (o recorte vai na evidência).
        entries: [(rotulo_veiculo, texto_placa)] do mais recente para o mais antigo."""
        if not entries:
            return
        H, W = img.shape[:2]
        rows = entries[:4]
        x0, y0, lh = W - 250, 84, 18
        cv2.rectangle(img, (x0 - 8, y0 - 4), (W - 6, y0 + 16 + lh * len(rows)), (0, 0, 0), -1)
        put_text(img, "PLACAS MONITORADAS", (x0, y0 + 10), BGR["branco"], 0.42, 1)
        for i, (veic, txt) in enumerate(rows):
            # ILEGIVEL: o sistema achou/recortou a placa, mas ela está borrada (privacidade) ou ilegível
            mostra = "XXX-XXXX (privada)" if txt == "ILEGIVEL" else txt
            cor = BGR["amarelo"] if txt == "ILEGIVEL" else BGR["verde"]
            put_text(img, f"{veic} - placa: {mostra}", (x0, y0 + 10 + lh * (i + 1)), cor, 0.45, 1)

    def add_alert(self, frame_idx, track, text):
        """Infração das regras (sinal vermelho, contramão, linha contínua...): rastro vermelho + etiqueta."""
        ev = {"frame": frame_idx, "track_id": track.id, "trilha": list(track.trail), "texto": text}
        self.items.append((frame_idx + self.hold, "alerta", ev))

    @staticmethod
    def draw_light(img, roi, state):
        """Contorna o semáforo na cor lida e escreve o estado (o que o sistema 'viu')."""
        if not roi:
            return
        x1, y1, x2, y2 = map(int, roi)
        cor = {"red": BGR["vermelho"], "yellow": BGR["amarelo"], "green": BGR["verde"]}.get(state, BGR["cinza"])
        nome = {"red": "VERMELHO", "yellow": "AMARELO", "green": "VERDE"}.get(state, "?")
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
        put_text(img, f"SINAL: {nome}", (x1, y2 + 22), cor, 0.65, 2)

    def draw(self, img, frame_idx, tracks):
        # rastro fino de todos os veículos (contexto)
        for t in tracks:
            draw_path(img, list(t.trail)[-30:], BGR["cinza"], 1, arrow=False)
        self.items = [it for it in self.items if it[0] >= frame_idx]
        self.active = [(kind, ev) for _, kind, ev in self.items]  # o que está na tela agora (nota de intensidade)
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
            elif kind == "ficaram":
                x, y = map(int, ev["ponto"])
                put_text(img, "FICARAM NO LOCAL", (x - 90, y + 48), BGR["verde"], 0.7, 2)
            elif kind == "alerta":
                t = live.get(ev["track_id"])
                pts = list(ev["trilha"]) + (list(t.trail)[-30:] if t is not None else [])
                draw_path(img, pts, BGR["vermelho"], 4)
                end = pts[-1]
                put_text(img, ev["texto"], (end[0] + 10, end[1] + 6), BGR["vermelho"], 0.8, 2)
            elif kind == "fuga":
                t = live.get(ev["track_id"])
                pts = list(ev["trilha"]) + (list(t.trail)[-30:] if t is not None else [])
                draw_path(img, pts, BGR["vermelho"], 4)
                end = pts[-1]
                txt = f"FUGA #{ev['track_id']}" + (" (roteiro)" if ev.get("fuga_do_roteiro") else "")
                put_text(img, txt, (end[0] + 8, end[1] + 18), BGR["vermelho"], 0.7, 2)
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
