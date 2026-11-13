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

from eda.geometry import point_in_polygon

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


def maneuver_from_trail(track, n_frames: int, tipo: str, frame_idx: int, reliable: bool):
    """Manobra marcada no roteiro (o QUANDO): rota = trajetória real dos últimos n_frames do track;
    lado = sinal do produto vetorial entre o rumo do começo e o do fim (geometria); cor = seta."""
    pts = list(track.trail)[-max(6, n_frames):]
    k = max(2, len(pts) // 4)
    d1 = _sub(_mean(pts[k:2 * k]), _mean(pts[:k]))
    d2 = _sub(_mean(pts[-k:]), _mean(pts[-2 * k:-k]))
    side = "direita" if _cross(d1, d2) > 0 else "esquerda"
    hist = track.state.get("mv_hist") or []
    labels = [e[1] for e in hist] + [track.state.get("blinker", "nenhuma")]
    cor, motivo = judge_signal(side, labels, reliable)
    return {"tipo": tipo, "lado": side, "cor": cor, "motivo": motivo, "track_id": track.id, "cls": track.cls,
            "frame": frame_idx, "pontos": pts, "giro_graus": round(_angle_deg(d1, d2), 1),
            "roteiro": True, "manobra_do_roteiro": True}


class ManeuverWatcher:
    """Um por câmera. update() por frame com os tracks confirmados; devolve eventos de manobra."""

    def __init__(self, fps: float, base_s: float = 1.0, lookback_s: float = 4.0, smooth: int = 5,
                 lateral_ratio: float = 1.0, turn_deg: float = 28.0, min_move_ratio: float = 0.6,
                 min_box_px: float = 0.0, frame_size=None, edge_frac: float = 0.03, settle_s: float = 3.0,
                 roi=None):
        self.fps = fps
        self.roi = roi  # região de fiscalização: fora dela a regra nem dispara (não gasta o tempo de espera)
        self.still_n = max(smooth, int(fps))  # "parado" medido em 1 s: carro devagar (conversão lenta) anda
        # a regra DETECTA no começo da curva, mas só DECIDE lado/tipo quando a manobra termina (settle_s depois
        # ou quando o veículo sai de cena): no começo o carro "abre" para o outro lado e engana o lado.
        self.settle = max(1, int(settle_s * fps))
        self.pending = {}  # track_id -> (track, frame do gatilho, frame inicial da manobra)
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
        x1, y1, x2, y2 = track.box
        blink = st.get("blinker", "nenhuma")
        raw = st.get("mv_raw")
        if raw is None:
            raw = st["mv_raw"] = deque(maxlen=self.still_n + 1)
        raw.append(track.point)
        # parado = andou menos de 30% da altura (piso 5 px) em 1 s. Em 5 frames, carro fazendo conversão
        # devagar contava como parado e a curva inteira sumia; carro parado de verdade só treme ±2 px.
        moving = not (len(raw) > self.still_n and math.hypot(track.point[0] - raw[0][0], track.point[1] - raw[0][1])
                      < max(0.3 * (y2 - y1), 5.0))
        if moving:  # caminho completo (sem os trechos parados), usado para decidir a manobra no fim
            path = st.get("mv_path")
            if path is None:
                path = st["mv_path"] = deque(maxlen=int(12 * self.fps))
            path.append((track.point, blink, frame_idx))
        if track.id in self.pending:  # manobra detectada: espera ela terminar para decidir o lado
            if frame_idx - self.pending[track.id][1] >= self.settle:
                return self._finalize(track.id, frame_idx, reliable)
            return None
        if frame_idx < st.get("mv_cooldown", -1):  # ainda terminando a manobra anterior
            hist.clear()
            return None
        if not moving:
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
        if self.roi and not point_in_polygon(track.point, self.roi):
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
        # gatilho: a manobra começou. Lado e tipo são decididos no fim (_finalize).
        self.pending[track.id] = (track, frame_idx, h[0][4], round(lateral, 1), width, track.point)
        hist.clear()
        st["mv_cooldown"] = frame_idx + self.settle + int(1.0 * self.fps)
        return None

    def _finalize(self, tid, frame_idx, reliable):
        """Decide a manobra olhando ela INTEIRA: rumo do começo x rumo do fim (como o olho faz)."""
        track, trig, start, lateral, width, gatilho = self.pending.pop(tid)
        st = track.state
        # trajetória INTEIRA andada até agora (sem os trechos parados): rumo de CHEGADA x rumo de SAÍDA.
        # Janelas curtas ficam reféns do tremor da caixa e da "abertura" antes da curva.
        seg = list(st.get("mv_path", []))
        if len(seg) < 8:
            return None
        pts = [e[0] for e in seg]
        k = max(3, len(pts) // 4)
        h1 = _sub(_mean(pts[k:2 * k]), _mean(pts[:k]))
        h2 = _sub(_mean(pts[-k:]), _mean(pts[-2 * k:-k]))
        turn = _angle_deg(h1, h2)
        if turn >= 10:  # girou: o lado é o do giro
            side = "direita" if _cross(h1, h2) > 0 else "esquerda"
        else:           # quase paralelo (troca de faixa): o lado é o do deslocamento
            side = "direita" if _cross(h1, _sub(pts[-1], pts[0])) > 0 else "esquerda"
        # câmera em diagonal "achata" a curva: conversão também é deslocamento lateral grande (troca de faixa
        # desloca ~1 a 1,5 largura do carro)
        kind = "conversao" if turn >= self.turn_deg or lateral >= 2.0 * width else "troca_faixa"
        labels = [e[1] for e in seg if e[2] >= start - self.maxlen]  # seta nos ~4 s antes da manobra
        parado = st.get("mv_blink_parado")
        if parado and frame_idx - parado[1] <= self.maxlen * 3:  # seta dada enquanto esperava parado
            labels.append(parado[0])
        cor, motivo = judge_signal(side, labels, reliable)
        return {"tipo": kind, "lado": side, "cor": cor, "motivo": motivo, "track_id": track.id,
                "cls": track.cls, "frame": frame_idx, "pontos": pts, "lateral_px": lateral,
                "giro_graus": round(turn, 1), "gatilho": gatilho, "_track": track}

    def flush(self, alive_ids, frame_idx, reliable_of=lambda t: False):
        """Fecha as manobras de veículos que saíram de cena (decide com o que foi visto)."""
        out = []
        for tid in [t for t in self.pending if t not in alive_ids]:
            track = self.pending[tid][0]
            ev = self._finalize(tid, frame_idx, reliable_of(track))
            if ev:
                out.append(ev)
        return out
