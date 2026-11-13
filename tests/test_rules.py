"""Testa regras e Detran simulado com trajetórias sintéticas (sem vídeo nem YOLO)."""
import json
import os

from detran_sim.scoring import DetranSim
from perception.detector import Detection
from perception.tracker import Tracker
from rules.engine import RuleEngine
from rules.state_machine import LaneChangeFSM

SCENE = os.path.join(os.path.dirname(__file__), "..", "data", "scenes", "exemplo_car_detection.json")
T0 = 1_780_000_000.0


def _drive(engine, points, light="red", cls="car"):
    tr, found = Tracker(min_hits=1), []
    for i, (x, y) in enumerate(points):
        box = (x - 20, y - 40, x + 20, y)
        tracks = tr.update([Detection(box, cls, 0.9)])
        found += engine.evaluate(tracks, i, T0 + i / 30, light)
    return {v.type for v in found}


def test_red_light_and_wrong_way():
    eng = RuleEngine(SCENE)
    down = [(200, 100 + 8 * i) for i in range(30)]  # desce cruzando a linha y=260
    types = _drive(eng, down, light="red")
    assert "avanco_sinal_vermelho" in types and "contramao" not in types
    up = [(200, 420 - 8 * i) for i in range(30)]    # sobe = contra o sentido (0, 1)
    assert "contramao" in _drive(eng, up, light="green")
    assert "avanco_sinal_vermelho" not in _drive(RuleEngine(SCENE), down, light="green")


def test_solid_line_and_illegal_turn():
    eng = RuleEngine(SCENE)
    right_to_left = [(500 - 10 * i, 200) for i in range(30)]
    types = _drive(eng, right_to_left, light="green")
    assert "transpor_linha_continua" in types and "conversao_proibida" in types


def test_lane_change_fsm():
    fsm = LaneChangeFSM("A")
    assert fsm.step(None, blinking=False) is None
    assert fsm.step("B", blinking=False) == ("lane_change", "A", "B", False)


def test_detran_three_leves_and_points():
    sim = DetranSim()
    base = {"camera_id": "c1", "track_id": 1, "plate": "ABC1D23", "severity": "leve"}
    acts = [sim.receive({**base, "type": f"t{i}", "timestamp": T0 + i * 3600})["acao"] for i in range(4)]
    assert acts[0] == "advertencia" and acts[3] == "multa_por_acumulo_de_leves"
    dup = sim.receive({**base, "type": "t3", "timestamp": T0 + 3 * 3600 + 10})
    assert dup["status"] == "duplicada"
    g = sim.receive({**base, "type": "sinal", "severity": "gravissima", "timestamp": T0 + 20000})
    assert g["pontos"] == 7 and g["pontos_12m"] == 10 and g["limite"] == 30


def test_gato_canteiro_e_re():
    eng = RuleEngine(SCENE)
    corta = [(20 + 5 * i, 405) for i in range(25)]          # atravessa o canteiro
    assert "gato_canteiro" in _drive(eng, corta, light="green")
    eng = RuleEngine(SCENE)
    vai = [(500, 100 + 6 * i) for i in range(20)]           # anda pra frente...
    para = [(500, 214)] * 5
    volta = [(500, 214 - 6 * i) for i in range(1, 20)]      # ...e volta em linha reta (ré)
    assert "gato_marcha_re" in _drive(eng, vai + para + volta, light="green")
    eng = RuleEngine(SCENE)
    reto = [(500, 50 + 6 * i) for i in range(50)]
    assert "gato_marcha_re" not in _drive(eng, reto, light="green")


def _lane_scene(tmp_path):
    scene = {
        "camera_id": "teste_faixas", "frame_size": [400, 400],
        "zones": [{"id": "f1", "polygon": [[0, 0], [200, 0], [200, 400], [0, 400]], "direction": [0, 1]},
                  {"id": "f2", "polygon": [[200, 0], [400, 0], [400, 400], [200, 400]], "direction": [0, 1]}],
        "allowed_moves": [["f1", "f2"], ["f2", "f1"]],
        "lane_lines": [[[200, 0], [200, 400]]], "lanes": ["f1", "f2"], "lane_margin": 10,
    }
    p = tmp_path / "faixas.json"
    p.write_text(json.dumps(scene))
    return str(p)


def _drive_blinker(engine, points, blinker):
    tr, found = Tracker(min_hits=1), []
    for i, (x, y) in enumerate(points):
        tracks = tr.update([Detection((x - 20, y - 40, x + 20, y), "car", 0.9)])
        for t in tracks:
            t.state["blinker"] = blinker
        found += engine.evaluate(tracks, i, T0 + i / 30, "green")
    return {v.type for v in found}


def test_lane_change_without_signal(tmp_path):
    path = _lane_scene(tmp_path)
    diagonal = [(150 + 4 * i, 50 + 8 * i) for i in range(30)]  # f1 -> sobre a linha -> f2
    assert "troca_faixa_sem_seta" in _drive_blinker(RuleEngine(path), diagonal, "nenhuma")
    assert "troca_faixa_sem_seta" not in _drive_blinker(RuleEngine(path), diagonal, "direita")
    straight = [(100, 50 + 8 * i) for i in range(30)]
    assert "troca_faixa_sem_seta" not in _drive_blinker(RuleEngine(path), straight, "nenhuma")
