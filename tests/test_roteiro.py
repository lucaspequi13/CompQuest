"""Modo roteiro: seta anotada à mão muda a cor da manobra; batida anotada -> o sistema decide a fuga."""
from perception.detector import Detection
from perception.tracker import Tracker
from rules.collision import CollisionWatcher
from rules.maneuver import AMARELO, VERDE, VERMELHO, ManeuverWatcher
from rules.roteiro import Roteiro

FPS = 30.0


def _path():
    pts = [(200, 50 + 8 * i) for i in range(40)]
    pts += [(200 + 4 * i, 370 + 8 * i) for i in range(1, 16)]
    pts += [(260, 490 + 8 * i) for i in range(1, 21)]
    return pts


def _run(roteiro_data):
    rot = Roteiro(roteiro_data, FPS)
    tr, mw, evs = Tracker(min_hits=1), ManeuverWatcher(FPS), []
    for i, (x, y) in enumerate(_path()):
        tracks = tr.update([Detection((x - 20, y - 40, x + 20, y), "car", 0.9)])
        for t in tracks:
            t.state["blinker"], t.state["blinker_reliable"] = "nenhuma", False
        rot.apply(tracks, i)
        for t in tracks:
            ev = mw.update(t, i, t.state["blinker_reliable"])
            if ev:
                evs.append(ev)
    return evs, rot


def test_roteiro_seta_colors():
    # carro desce pela tela; em t=0,5 s está perto de (200, 170)
    ok, rot = _run({"seta": [{"t0": 0.5, "t1": 2.0, "x": 200, "y": 160, "lado": "esquerda"}]})
    assert ok[0]["cor"] == VERDE and rot.log[0]["track_id"] == 1
    sem, _ = _run({"seta": [{"t0": 0.5, "t1": 2.0, "x": 200, "y": 160, "lado": "nenhuma"}]})
    assert sem[0]["cor"] == VERMELHO
    nada, _ = _run({})
    assert nada[0]["cor"] == AMARELO
    longe, rot = _run({"seta": [{"t0": 0.5, "t1": 2.0, "x": 600, "y": 50, "lado": "esquerda"}]})
    assert longe[0]["cor"] == AMARELO and rot.seta[0]["track"] is None   # não achou veículo: não inventa


def test_roteiro_batida_then_system_decides_fuga():
    rot = Roteiro({"batida": [{"t": 0.2, "x": 300, "y": 190}]}, FPS)
    tr, cw, evs = Tracker(min_hits=1), CollisionWatcher(FPS, (640, 480)), []
    a = [(270, 200)] * 10 + [(270, 200 - 12 * i) for i in range(1, 30)]     # A sai depois da batida
    for i, pa in enumerate(a):
        dets = [Detection((300, 160, 360, 200), "car", 0.9), Detection((pa[0] - 20, pa[1] - 40, pa[0] + 20, pa[1]), "car", 0.9)]
        tracks = tr.update(dets)
        for ida, idb, p, lv in rot.apply(tracks, i):
            evs.append(cw.force(ida, idb, p, i, lv))
        evs += cw.update(tracks, {t.id for t in tr.tracks}, i)
    fugas = [e for e in evs if e["tipo"] == "fuga"]
    assert any(e.get("roteiro") for e in evs if e["tipo"] == "colisao")
    assert len(fugas) == 1 and fugas[0]["track_id"] == 2 and fugas[0]["ficou"] == [1]
    assert fugas[0]["roteiro"] is True and not fugas[0].get("fuga_do_roteiro")  # fuga decidida pelo sistema


def test_roteiro_fuga_marcada_quando_sistema_nao_ve():
    """Vítima some da detecção (capotou) e o outro vai embora devagar em profundidade: o roteiro diz quem fugiu."""
    rot = Roteiro({"batida": [{"t": 0.2, "x": 330, "y": 190, "x2": 270, "y2": 195, "fugiu": 2}]}, FPS)
    tr, cw, evs = Tracker(min_hits=1), CollisionWatcher(FPS, (640, 480)), []
    for i in range(80):
        dets = [Detection((250, 160, 290, 200), "car", 0.9)]               # quem foge (anda pouco na tela)
        if i < 12:
            dets.append(Detection((300, 160, 360, 200), "car", 0.9))       # vítima some depois
        tracks = tr.update(dets)
        for ida, idb, p, lv in rot.apply(tracks, i):
            evs.append(cw.force(ida, idb, p, i, lv))
        evs += cw.update(tracks, {t.id for t in tr.tracks}, i)
    fugas = [e for e in evs if e["tipo"] == "fuga"]
    assert len(fugas) == 1 and fugas[0]["fuga_do_roteiro"] is True
    assert fugas[0]["track_id"] == rot.log[0]["fugiu"]
