"""Manobra (verde/vermelho/amarelo) e batida com fuga, com trajetórias sintéticas."""
from perception.detector import Detection
from perception.tracker import Tracker
from rules.collision import CollisionWatcher
from rules.maneuver import AMARELO, VERDE, VERMELHO, ManeuverWatcher


def _lane_change_path():
    pts = [(200, 50 + 8 * i) for i in range(40)]                         # desce reto (vem para a câmera)
    pts += [(200 + 4 * i, 370 + 8 * i) for i in range(1, 16)]             # vai 60 px para a direita da tela
    pts += [(260, 490 + 8 * i) for i in range(1, 21)]                      # segue reto
    return pts


def _maneuvers(points, blinker="nenhuma", reliable=False):
    tr, mw, evs = Tracker(min_hits=1), ManeuverWatcher(30.0), []
    for i, (x, y) in enumerate(points):
        for t in tr.update([Detection((x - 20, y - 40, x + 20, y), "car", 0.9)]):
            t.state["blinker"] = blinker
            ev = mw.update(t, i, reliable)
            if ev:
                evs.append(ev)
    return evs


def test_maneuver_side_and_colors():
    path = _lane_change_path()
    evs = _maneuvers(path)
    assert len(evs) == 1
    # descendo na tela = vindo para a câmera; ir para a direita da tela = esquerda do motorista
    assert evs[0]["lado"] == "esquerda" and evs[0]["tipo"] == "troca_faixa"
    assert evs[0]["cor"] == AMARELO                                        # seta não confiável: não acusa
    assert _maneuvers(path, "nenhuma", reliable=True)[0]["cor"] == VERMELHO
    assert _maneuvers(path, "direita", reliable=True)[0]["cor"] == VERMELHO  # lado errado
    assert _maneuvers(path, "esquerda", reliable=True)[0]["cor"] == VERDE
    assert _maneuvers(path, "esquerda", reliable=False)[0]["cor"] == VERDE


def test_stop_then_turn_is_detected():
    """Para no cruzamento esperando e depois vira: o rumo de referência é o de antes de parar? Não:
    recomeça quando volta a andar, então precisa andar um pouco reto e virar (caso típico de fila)."""
    pts = [(200, 50 + 8 * i) for i in range(30)]                  # desce
    pts += [(200, 282)] * 90                                       # 3 s parado no sinal
    pts += [(200, 282 + 8 * i) for i in range(1, 40)]              # arranca reto
    pts += [(200 + 8 * i, 594) for i in range(1, 40)]              # vira para a direita da tela
    evs = _maneuvers(pts, "esquerda", reliable=True)
    assert len(evs) == 1 and evs[0]["tipo"] == "conversao" and evs[0]["lado"] == "esquerda"
    assert evs[0]["cor"] == VERDE


def test_blinker_while_stopped_counts():
    pts = [(200, 50 + 8 * i) for i in range(30)] + [(200, 282)] * 90
    pts += [(200, 282 + 8 * i) for i in range(1, 40)] + [(200 + 8 * i, 594) for i in range(1, 40)]
    tr, mw, evs = Tracker(min_hits=1), ManeuverWatcher(30.0), []
    for i, (x, y) in enumerate(pts):
        for t in tr.update([Detection((x - 20, y - 40, x + 20, y), "car", 0.9)]):
            t.state["blinker"] = "esquerda" if 60 <= i < 120 else "nenhuma"   # piscou só parado
            ev = mw.update(t, i, True)
            if ev:
                evs.append(ev)
    assert evs and evs[0]["cor"] == VERDE


def test_straight_has_no_maneuver():
    assert _maneuvers([(200, 50 + 8 * i) for i in range(80)]) == []


def _crash(a_after):
    """B parado em (300, 200). A vem da esquerda, bate em B (para de repente) e depois faz `a_after`."""
    a = [(100 + 12 * i, 200) for i in range(17)] + [(292, 200)] * 20 + a_after
    tr, cw, evs = Tracker(min_hits=1), CollisionWatcher(30.0, (640, 480)), []
    for i, pa in enumerate(a):
        dets = [Detection((270, 160, 330, 200), "car", 0.9)]
        if pa is not None:
            dets.append(Detection((pa[0] - 20, pa[1] - 40, pa[0] + 20, pa[1]), "car", 0.9))
        tracks = tr.update(dets)
        evs += cw.update(tracks, {t.id for t in tr.tracks}, i)
    return evs


def test_collision_and_hit_and_run():
    fugiu = [(292, 200 - 12 * i) for i in range(1, 30)]                    # A vai embora para cima
    evs = _crash(fugiu)
    kinds = [e["tipo"] for e in evs]
    assert kinds.count("colisao") == 1 and kinds.count("fuga") == 1
    fuga = next(e for e in evs if e["tipo"] == "fuga")
    assert fuga["track_id"] != 1 and fuga["ficou"] == [1]                 # B (id 1) ficou


def test_collision_without_leaving_is_not_hit_and_run():
    evs = _crash([(292, 200)] * 150)                                       # A fica parado no local
    kinds = [e["tipo"] for e in evs]
    assert "colisao" in kinds and "fuga" not in kinds
