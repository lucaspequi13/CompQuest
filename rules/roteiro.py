"""Modo roteiro: anotações à mão para a demo, por cima da detecção e do tracking REAIS.

O que vem do roteiro (e aparece marcado como "roteiro" no vídeo e nos JSONs):
  - estado da SETA de um veículo num intervalo de tempo (o classificador de seta ainda não funciona);
  - o MOMENTO de uma batida (evento raro, difícil de pegar em vídeo).
O resto é do sistema: quem é o veículo (YOLO + tracker), se ele trocou de faixa/converteu e para que lado
(ManeuverWatcher), se a manobra teve seta (comparando com o roteiro), quem foi embora depois da batida
(CollisionWatcher).

Formato (tempos em segundos desde o início do vídeo; x, y em pixels do vídeo original):
{
  "video": "data/videos/xxx.mp4",
  "seta": [
    {"t0": 2.0, "t1": 6.0, "x": 640, "y": 400, "lado": "esquerda"},   # deu seta para a esquerda
    {"t0": 9.0, "t1": 13.0, "x": 300, "y": 500, "lado": "nenhuma"}    # conferido: NÃO deu seta
  ],
  "batida": [
    {"t": 15.2, "x": 420, "y": 380},                                  # os 2 veículos mais perto do ponto
    {"t": 20.0, "x": 300, "y": 200, "x2": 340, "y2": 210},            # ou um ponto em cada veículo
    {"t": 3.7, "x": 595, "y": 160, "x2": 675, "y2": 200, "fugiu": 2}, # fuga marcada à mão (2º veículo)
    {"t": 2.2, "x": 560, "y": 330, "vitima": [340, 300]}              # veículo x vítima não rastreada
  ]
  "manobra": [
    {"t0": 0.4, "t1": 2.0, "x": 406, "y": 290, "tipo": "troca_faixa"} # quando a geometria não confirma sozinha
  ]
}
"vitima" = onde a moto/pedestre caiu (o YOLO não pega moto pequena/caída). A vítima fica no local por definição;
se o veículo for embora, o sistema acusa a fuga sozinho.
Sem "fugiu", quem decide a fuga é o sistema (um se afasta, o outro fica).
"manobra" = o roteiro diz QUANDO houve a manobra (ex.: troca de faixa que acontece "em profundidade" na
perspectiva da câmera); a rota desenhada é a trajetória real, o LADO vem da geometria e a cor vem da seta.
O veículo é achado pela posição no instante t0/t: a caixa que contém (x, y), ou a mais perto (até max_dist px).
Só veículos citados no roteiro têm a seta "confiável"; os demais continuam amarelos (revisão).

Gerar o arquivo clicando no vídeo: python scripts/roteiro_editor.py --video ...
"""
import json
import math

SIDES = ("esquerda", "direita", "nenhuma")


def _dist_to_box(x, y, box):
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return math.hypot(dx, dy)


def nearest_tracks(tracks, x, y, n=1, max_dist=80.0, exclude=()):
    cand = sorted((_dist_to_box(x, y, t.box), t.id, t) for t in tracks if t.id not in exclude)
    return [t for d, _, t in cand if d <= max_dist][:n]


class Roteiro:
    def __init__(self, data: dict, fps: float, max_dist: float = 80.0):
        self.fps, self.max_dist = fps, max_dist
        self.seta = []
        for s in data.get("seta", []):
            if s["lado"] not in SIDES:
                raise ValueError(f"lado inválido no roteiro: {s['lado']} (use {SIDES})")
            self.seta.append({**s, "f0": round(s["t0"] * fps), "f1": round(s["t1"] * fps), "track": None})
        self.batida = [{**b, "f": round(b["t"] * fps), "done": False} for b in data.get("batida", [])]
        # manobra que a geometria não confirma sozinha (troca de faixa "em profundidade"): o roteiro diz QUANDO;
        # a rota desenhada é a trajetória real do track e o LADO é calculado pela geometria.
        self.manobra = [{**m, "f0": round(m["t0"] * fps), "f1": round(m["t1"] * fps), "done": False}
                        for m in data.get("manobra", [])]
        self.log = []  # o que foi aplicado, para conferir

    @classmethod
    def load(cls, path: str, fps: float, **kw):
        with open(path, encoding="utf-8-sig") as f:  # aceita arquivo com BOM (Bloco de Notas / PowerShell)
            return cls(json.load(f), fps, **kw)

    def apply(self, tracks, frame_idx: int):
        """Chamar depois do classificador de seta. Sobrescreve a seta dos veículos do roteiro e
        devolve as batidas que começam neste frame: [(id_a, id_b, (x, y))]."""
        for s in self.seta:
            if s["track"] is None and s["f0"] <= frame_idx <= s["f1"]:
                hit = nearest_tracks(tracks, s["x"], s["y"], 1, self.max_dist)
                if hit:
                    s["track"] = hit[0].id
                    self.log.append({"frame": frame_idx, "seta": s["lado"], "track_id": s["track"]})
        bound = {s["track"] for s in self.seta if s["track"] is not None}
        for t in tracks:
            if t.id not in bound:
                continue
            t.state["roteiro"] = True
            t.state["blinker_reliable"] = True        # o roteiro diz a verdade sobre a seta deste veículo
            now = [s["lado"] for s in self.seta if s["track"] == t.id and s["f0"] <= frame_idx <= s["f1"]]
            t.state["blinker"] = now[0] if now else "nenhuma"  # fora dos intervalos anotados: sem seta
        forced = []
        for b in self.batida:
            if not b["done"] and frame_idx >= b["f"]:
                if "vitima" in b:  # vítima não rastreada (moto/pedestre caído): ponto fixo; o sistema decide a fuga
                    hit = nearest_tracks(tracks, b["x"], b["y"], 1, self.max_dist)
                    if hit:
                        b["done"] = True
                        forced.append((hit[0].id, None, tuple(b["vitima"]), None))
                        self.log.append({"frame": frame_idx, "batida": [hit[0].id, "vitima"]})
                    elif frame_idx > b["f"] + self.fps:
                        b["done"] = True
                        self.log.append({"frame": frame_idx, "batida_nao_achada": [b["x"], b["y"]]})
                    continue
                if "x2" in b:  # um ponto em cada veículo: sem ambiguidade em trânsito cheio
                    first = nearest_tracks(tracks, b["x"], b["y"], 1, self.max_dist)
                    second = nearest_tracks(tracks, b["x2"], b["y2"], 1, self.max_dist,
                                            exclude={t.id for t in first})
                    pair = first + second
                else:
                    pair = nearest_tracks(tracks, b["x"], b["y"], 2, self.max_dist * 2)
                if len(pair) == 2:
                    b["done"] = True
                    # "fugiu": 1 ou 2 = qual dos pontos é quem foi embora (quando o sistema não consegue ver,
                    # ex.: vítima capotada some da detecção, fuga em profundidade no túnel). Fica marcado como roteiro.
                    leaver = pair[b["fugiu"] - 1].id if b.get("fugiu") in (1, 2) and "x2" in b else None
                    forced.append((pair[0].id, pair[1].id, (b["x"], b["y"]), leaver))
                    self.log.append({"frame": frame_idx, "batida": [pair[0].id, pair[1].id], "fugiu": leaver})
                elif frame_idx > b["f"] + self.fps:  # 1 s sem achar 2 veículos: desiste e avisa
                    b["done"] = True
                    self.log.append({"frame": frame_idx, "batida_nao_achada": [b["x"], b["y"]]})
        return forced

    def maneuvers(self, tracks, frame_idx: int):
        """Manobras do roteiro que terminam neste frame: [(track, frames_desde_o_inicio, tipo)]."""
        out = []
        for m in self.manobra:
            if not m["done"] and frame_idx >= m["f1"]:
                hit = nearest_tracks(tracks, m["x"], m["y"], 1, self.max_dist)
                if hit:
                    m["done"] = True
                    out.append((hit[0], frame_idx - m["f0"], m.get("tipo", "troca_faixa")))
                    self.log.append({"frame": frame_idx, "manobra": m.get("tipo", "troca_faixa"),
                                     "track_id": hit[0].id})
                elif frame_idx > m["f1"] + self.fps:
                    m["done"] = True
                    self.log.append({"frame": frame_idx, "manobra_nao_achada": [m["x"], m["y"]]})
        return out
