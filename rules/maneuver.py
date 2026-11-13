"""Manobra (troca de faixa / conversão) detectada pela TRAJETÓRIA, sem precisar marcar faixas na cena,
e a cor da manobra pela seta vista antes dela (CTB art. 196: indicar com antecedência).

Cores (não induzir: só acusa com certeza):
  verde    = seta para o lado da manobra
  vermelho = sem seta, ou seta do lado errado, E o classificador de seta é confiável -> infração
  amarelo  = classificador de seta não confiável -> revisão humana, não vira multa

Geometria (EDA): rumo de referência d (início da janela) e deslocamento v depois dele.
  lateral = |d x v| / |d|  (produto vetorial = distância à reta do rumo)
  lado    = sinal de d x v. Com y para baixo (imagem), d x v > 0 = à DIREITA do motorista.
Em perspectiva, reta da via vira reta na imagem, então andar reto não gera deslocamento lateral.
"""
import math
from collections import deque

VERDE, VERMELHO, AMARELO = "verde", "vermelho", "amarelo"
SIDES = ("esquerda", "direita")


def _mean(pts):
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _angle_deg(u, v):
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(c))


def judge_signal(side, labels, reliable):
    """Cor e motivo a partir dos rótulos de seta vistos na janela."""
    seen = {s for s in labels if s in SIDES}
    if side in seen:
        return VERDE, f"seta {side} ok"
    if not reliable:
        return AMARELO, "seta incerta: revisão humana"
    if seen:
        return VERMELHO, f"seta do lado errado ({'/'.join(sorted(seen))})"
    return VERMELHO, "sem seta"


class ManeuverWatcher:
    """Um por câmera. update() por frame com os tracks confirmados; devolve eventos de manobra."""

    def __init__(self, fps: float, base_s: float = 1.0, lookback_s: float = 4.0, smooth: int = 5,
                 lateral_ratio: float = 1.0, turn_deg: float = 28.0, min_move_ratio: float = 0.6,
                 min_box_px: float = 0.0, frame_size=None, edge_frac: float = 0.03):
        self.fps = fps
        self.frame_size = frame_size                         # (w, h): ignora manobra colada na borda
        self.edge_frac = edge_frac
        self.base = max(smooth + 1, int(base_s * fps))       # frames que definem o rumo de referência
        self.maxlen = max(self.base * 2, int(lookback_s * fps))
        self.smooth = smooth
        self.lateral_ratio = lateral_ratio                   # deslocamento lateral em larguras de caixa
        self.turn_deg = turn_deg
        self.min_move_ratio = min_move_ratio                 # andar ao menos isso (em alturas) no rumo base
        self.min_box_px = min_box_px                         # carro pequeno (longe) = trajetória tremida

    def update(self, track, frame_idx: int, reliable: bool):
        st = track.state
        hist = st.get("mv_hist")
        if hist is None:
            hist = st["mv_hist"] = deque(maxlen=self.maxlen)
        if frame_idx < st.get("mv_cooldown", -1):  # ainda terminando a manobra anterior
            hist.clear()
            return None
        x1, y1, x2, y2 = track.box
        blink = st.get("blinker", "nenhuma")
        raw = st.get("mv_raw")
        if raw is None:
            raw = st["mv_raw"] = deque(maxlen=self.smooth + 1)
        raw.append(track.point)
        if len(raw) > self.smooth and math.hypot(track.point[0] - raw[0][0], track.point[1] - raw[0][1]) \
                < max(0.1 * (y2 - y1), 4.0):  # piso em px: em carro pequeno o tremor da caixa já passa de 10%
            # parado (fila, sinal): não entra no histórico. A trajetória de antes de parar emenda com a de
            # depois, então o rumo de referência é "de onde ele vinha" (certo para quem para e vira).
            # A seta vista parado conta: é aí que o motorista costuma ligar o pisca.
            if blink in ("esquerda", "direita"):
                st["mv_blink_parado"] = (blink, frame_idx)
            return None
        hist.append((track.point, blink, x2 - x1, y2 - y1, frame_idx))
        if len(hist) < self.base + self.smooth or (y2 - y1) < self.min_box_px:
            return None
        if self.frame_size:
            W, H = self.frame_size
            mx, my = self.edge_frac * W, self.edge_frac * H
            if x1 < mx or y1 < my or x2 > W - mx or y2 > H - my:  # caixa cortada / rua curvando na borda
                return None
        h = list(hist)
        k = self.smooth
        p0 = _mean([e[0] for e in h[:k]])
        p1 = _mean([e[0] for e in h[self.base - k:self.base]])
        pc = _mean([e[0] for e in h[-k:]])
        width = sum(e[2] for e in h) / len(h)
        height = sum(e[3] for e in h) / len(h)
        d = _sub(p1, p0)
        if math.hypot(*d) < self.min_move_ratio * height:  # parado ou quase: sem rumo confiável
            return None
        v = _sub(pc, p1)
        lateral = abs(_cross(d, v)) / math.hypot(*d)
        recent = _sub(pc, _mean([e[0] for e in h[-self.base:-self.base + k]]))
        m = max(k, int(0.3 * self.fps))                 # rumo instantâneo: já girou no começo da curva
        inst = _sub(pc, _mean([e[0] for e in h[-m - k:-m]]))
        turn = max(_angle_deg(d, recent), _angle_deg(d, inst))
        if lateral < 0.5 * width:  # sem deslocamento lateral de verdade: tremor, não manobra
            return None
        if lateral < self.lateral_ratio * width and turn < self.turn_deg:
            return None
        side = "direita" if _cross(d, v) > 0 else "esquerda"
        kind = "conversao" if turn >= self.turn_deg else "troca_faixa"
        labels = [e[1] for e in h]
        parado = st.get("mv_blink_parado")
        if parado and frame_idx - parado[1] <= self.maxlen * 2:  # seta dada enquanto esperava parado
            labels.append(parado[0])
        cor, motivo = judge_signal(side, labels, reliable)
        ev = {"tipo": kind, "lado": side, "cor": cor, "motivo": motivo, "track_id": track.id,
              "cls": track.cls, "frame": frame_idx, "pontos": [e[0] for e in h],
              "lateral_px": round(lateral, 1), "giro_graus": round(turn, 1)}
        # recomeça depois de 1,5 s: a próxima manobra precisa de um rumo novo
        hist.clear()
        st["mv_cooldown"] = frame_idx + int(1.5 * self.fps)
        return ev
