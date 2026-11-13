"""Uma função por infração. Cada uma recebe (track, ctx) e devolve Violation ou None.

ctx tem: scene (dict do JSON), frame_idx, timestamp, light ("red"/...), graph (RoadGraph),
hour (int), make(type, severity, ctb, track) -> Violation.
Cada infração é reportada no máximo uma vez por track (track.state["reported"]).
"""
from eda.geometry import cos_angle, point_in_polygon, point_segment_distance, segments_intersect
from rules.state_machine import LaneChangeFSM


def _once(track, key) -> bool:
    rep = track.state.setdefault("reported", set())
    if key in rep:
        return False
    rep.add(key)
    return True


def _last_move(track):
    if len(track.trail) < 2:
        return None
    return track.trail[-2], track.trail[-1]


def red_light(track, ctx):
    """Art. 208 — avançar o sinal vermelho. Gravíssima."""
    line = ctx.scene.get("stop_line")
    mv = _last_move(track)
    if not line or not mv or ctx.light != "red":
        return None
    if segments_intersect(mv[0], mv[1], tuple(line[0]), tuple(line[1])) and _once(track, "red_light"):
        return ctx.make("avanco_sinal_vermelho", "gravissima", "Art. 208", track)


def wrong_way(track, ctx, min_frames: int = 10, min_speed: float = 1.0):
    """Art. 186 — contramão. Vetor de velocidade oposto ao sentido da zona por N frames."""
    for z in ctx.scene.get("zones", []):
        if not z.get("direction") or not point_in_polygon(track.point, z["polygon"]):
            continue
        vx, vy = track.velocity
        if (vx * vx + vy * vy) ** 0.5 < min_speed:
            return None
        if cos_angle((vx, vy), tuple(z["direction"])) < -0.5:
            track.state["ww"] = track.state.get("ww", 0) + 1
        else:
            track.state["ww"] = 0
        if track.state["ww"] >= min_frames and _once(track, "wrong_way"):
            sev = "gravissima" if z.get("one_way") else "grave"
            return ctx.make("contramao", sev, "Art. 186", track)
    return None


def crosswalk_stop(track, ctx, seconds: float = 2.0, fps: float = 30.0):
    """Art. 183 — parar sobre a faixa de pedestres. Média."""
    poly = ctx.scene.get("crosswalk")
    if not poly:
        return None
    vx, vy = track.velocity
    stopped = (vx * vx + vy * vy) ** 0.5 < 0.5
    if point_in_polygon(track.point, poly) and stopped:
        track.state["cw"] = track.state.get("cw", 0) + 1
    else:
        track.state["cw"] = 0
    if track.state["cw"] >= seconds * fps and _once(track, "crosswalk"):
        return ctx.make("parada_sobre_faixa_pedestre", "media", "Art. 183", track)


def bus_lane(track, ctx):
    """Art. 184 — transitar na faixa exclusiva de ônibus no horário regulamentado. Gravíssima."""
    bl = ctx.scene.get("bus_lane")
    if not bl or track.cls in ("bus",):
        return None
    if not any(a <= ctx.hour < b for a, b in bl.get("hours", [[0, 24]])):
        return None
    if point_in_polygon(track.point, bl["polygon"]):
        track.state["bl"] = track.state.get("bl", 0) + 1
        if track.state["bl"] >= 15 and _once(track, "bus_lane"):
            return ctx.make("faixa_exclusiva_onibus", "gravissima", "Art. 184", track)


def solid_line(track, ctx):
    """Arts. 185/203 — transpor linha contínua."""
    mv = _last_move(track)
    if not mv:
        return None
    for i, seg in enumerate(ctx.scene.get("solid_lines", [])):
        if segments_intersect(mv[0], mv[1], tuple(seg[0]), tuple(seg[1])) and _once(track, f"solid{i}"):
            return ctx.make("transpor_linha_continua", "media", "Art. 185", track)


def illegal_turn(track, ctx):
    """Arts. 206/207 — conversão/retorno proibido: transição de zona sem aresta no grafo."""
    zone = None
    for z in ctx.scene.get("zones", []):
        if point_in_polygon(track.point, z["polygon"]):
            zone = z["id"]
            break
    if zone and (not track.zones or track.zones[-1] != zone):
        track.zones.append(zone)
        bad = ctx.graph.first_illegal_move(track.zones[-2:]) if len(track.zones) >= 2 else None
        if bad and _once(track, f"turn{bad}"):
            v = ctx.make("conversao_proibida", "grave", "Art. 207", track)
            v.extra["move"] = list(bad)
            return v


def _current_lane(track, ctx):
    """Faixa onde o veículo está; None se estiver sobre uma linha tracejada (cruzando)."""
    margin = ctx.scene.get("lane_margin", 8)
    for a, b in ctx.scene.get("lane_lines", []):
        if point_segment_distance(track.point, tuple(a), tuple(b)) < margin:
            return None
    lanes = set(ctx.scene.get("lanes", []))
    for z in ctx.scene.get("zones", []):
        if z["id"] in lanes and point_in_polygon(track.point, z["polygon"]):
            return z["id"]
    return "fora"


def lane_change_no_signal(track, ctx):
    """Art. 196 — trocar de faixa sem dar seta. Grave.
    track.state["blinker"] vem do classificador de seta (perception/blinker.py)."""
    if not ctx.scene.get("lanes"):
        return None
    lane = _current_lane(track, ctx)
    fsm = track.state.get("lane_fsm")
    if fsm is None:
        if lane in (None, "fora"):
            return None
        fsm = track.state["lane_fsm"] = LaneChangeFSM(lane)
    if lane == "fora":
        return None
    blinking = track.state.get("blinker") in ("esquerda", "direita")
    ev = fsm.step(lane, blinking)
    # Não induzir: sem classificador de seta confiável, não acusa (a manobra vai para revisão no pipeline)
    if ev and not ev[3] and track.state.get("blinker_reliable", True) and _once(track, f"lane{ev[1]}>{ev[2]}"):
        v = ctx.make("troca_faixa_sem_seta", "grave", "Art. 196", track)
        v.extra["move"] = [ev[1], ev[2]]
        return v


# ---------------------------------------------------------------------------
# "GATO": atalho irregular — cortar caminho por onde não é via.
# Na cena, "forbidden_areas" = polígonos de canteiro, ilha, calçada, terra entre pistas.
# ---------------------------------------------------------------------------

def gato_canteiro(track, ctx, min_frames: int = 5):
    """Arts. 193 / 206 — veículo passando por canteiro, ilha, calçada ou terra
    para fazer retorno forçado ou trocar de pista. Gravíssima."""
    for i, area in enumerate(ctx.scene.get("forbidden_areas", [])):
        poly = area["polygon"] if isinstance(area, dict) else area
        if point_in_polygon(track.point, poly):
            key = f"fa{i}"
            track.state[key] = track.state.get(key, 0) + 1
            if track.state[key] >= min_frames and _once(track, f"gato_canteiro{i}"):
                v = ctx.make("gato_canteiro", "gravissima", "Arts. 193/206", track)
                v.extra["area"] = area.get("tipo", "canteiro") if isinstance(area, dict) else "canteiro"
                return v
        else:
            track.state[f"fa{i}"] = 0
    return None


def gato_re(track, ctx, min_px: float = 25.0):
    """Art. 194 — marcha à ré fora de pequena manobra (ex.: dá ré no sinal para
    sair por outra rua). Detecção: o veículo vinha num sentido, para e volta no
    sentido OPOSTO quase em linha reta (sem o arco de um retorno)."""
    tr = list(track.trail)
    if len(tr) < 30:
        return None
    a, b, c = tr[-30], tr[-15], tr[-1]
    fwd = (b[0] - a[0], b[1] - a[1])
    back = (c[0] - b[0], c[1] - b[1])
    n1, n2 = (fwd[0] ** 2 + fwd[1] ** 2) ** 0.5, (back[0] ** 2 + back[1] ** 2) ** 0.5
    if n1 < min_px or n2 < min_px:
        return None
    if cos_angle(fwd, back) < -0.85 and _once(track, "gato_re"):
        v = ctx.make("gato_marcha_re", "grave", "Art. 194", track)
        v.extra["deslocamento_px"] = round(n2, 1)
        return v
    return None


ALL_CHECKS = [red_light, wrong_way, crosswalk_stop, bus_lane, solid_line, illegal_turn, lane_change_no_signal, gato_canteiro, gato_re]
