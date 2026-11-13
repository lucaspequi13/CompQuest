"""Pipeline ponta a ponta sobre um vídeo.

python scripts/run_pipeline.py --video data/videos/car-detection.mp4 \
    --scene data/scenes/exemplo_car_detection.json --out runs/demo

Saídas em --out: video_anotado.mp4, infracoes.jsonl, decisoes.jsonl, revisao.jsonl e evidence/
Vídeo: só veículos (--all-classes para ver tudo); rastro verde/vermelho/amarelo nas manobras; BATIDA / FUGA.
"""
import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detran_sim.scoring import DetranSim  # noqa: E402
from eda.geometry import point_in_polygon  # noqa: E402
from eda.priority_heap import ViolationQueue  # noqa: E402
from eda.ring_buffer import RingBuffer  # noqa: E402
from evidence.packager import EvidencePackager  # noqa: E402
from perception.blinker import BlinkerClassifier  # noqa: E402
from dashboard.overlay import Overlay  # noqa: E402
from perception.detector import VEHICLES, Detector  # noqa: E402
from perception.tracker import Tracker  # noqa: E402
from perception.traffic_light import light_state  # noqa: E402
from rules.collision import CollisionWatcher  # noqa: E402
from rules.engine import RuleEngine  # noqa: E402
from rules.maneuver import VERMELHO, ManeuverWatcher  # noqa: E402
from rules.roteiro import Roteiro  # noqa: E402
from rules.types import Violation  # noqa: E402

MANOBRA_INFRACAO = {"troca_faixa": "troca_faixa_sem_seta", "conversao": "conversao_sem_seta"}


def draw_scene(frame, scene):
    import numpy as np
    for z in scene.get("zones", []):
        cv2.polylines(frame, [np.array(z["polygon"], int)], True, (255, 200, 0), 1)
        cx, cy = map(int, np.mean(z["polygon"], axis=0))
        cv2.putText(frame, z["id"], (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
    if scene.get("stop_line"):
        a, b = scene["stop_line"]
        cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (0, 0, 255), 2)
    for a, b in scene.get("lane_lines", []):
        cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (200, 200, 200), 1)
    for a, b in scene.get("solid_lines", []):
        cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 2)
    if scene.get("crosswalk"):
        cv2.polylines(frame, [np.array(scene["crosswalk"], int)], True, (255, 255, 255), 1)
    if scene.get("bus_lane"):
        cv2.polylines(frame, [np.array(scene["bus_lane"]["polygon"], int)], True, (0, 128, 255), 1)
    if scene.get("maneuver_roi"):
        cv2.polylines(frame, [np.array(scene["maneuver_roi"], int)], True, (120, 120, 120), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--weights", default="models/pretrained/yolo11n.pt")
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--light", default="auto", help="auto (usa ROI da cena) | red | green | unknown")
    ap.add_argument("--blinker-model", default="models/blinker_rf.joblib",
                    help="RandomForest de scripts/train_blinker.py (sem arquivo = baseline heurístico)")
    ap.add_argument("--roteiro", help="JSON com seta/batida anotadas à mão (modo roteiro, ver rules/roteiro.py)")
    ap.add_argument("--min-box", type=float, default=0.06,
                    help="altura mínima da caixa (fração da imagem) p/ manobra e batida. Vista de cima/drone: 0.02")
    ap.add_argument("--dump-tracks", action="store_true", help="grava tracks.csv (frame, id, classe, caixa)")
    ap.add_argument("--all-classes", action="store_true", help="mostrar também pedestre/bicicleta")
    ap.add_argument("--start-time", type=float, default=time.time(), help="epoch do 1º frame")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(os.path.join(args.out, "video_anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    det = Detector(args.weights, keep=None if args.all_classes else VEHICLES)
    trk, eng = Tracker(), RuleEngine(args.scene)
    blink = BlinkerClassifier(fps, args.blinker_model)
    print("Seta:", "RandomForest" if blink.model else "baseline heurístico (sem modelo treinado)",
          "| acusa sem seta:", "sim" if blink.reliable else "não (manobras vão para revisão, amarelo)")
    mw = ManeuverWatcher(fps, min_box_px=args.min_box * h, frame_size=(w, h))
    cw = CollisionWatcher(fps, (w, h), min_box_frac=args.min_box)
    ov = Overlay(fps)
    lanes_in_scene = bool(eng.scene.get("lanes"))
    rot = Roteiro.load(args.roteiro, fps) if args.roteiro else None
    if rot:
        ov.roteiro = True
        print(f"MODO ROTEIRO: {len(rot.seta)} seta(s), {len(rot.batida)} batida(s) anotadas à mão")
    f_rev = open(os.path.join(args.out, "revisao.jsonl"), "w")
    f_trk = open(os.path.join(args.out, "tracks.csv"), "w") if args.dump_tracks else None
    if f_trk:
        f_trk.write("frame,t,id,cls,x1,y1,x2,y2\n")

    def make(typ, sev, ctb, tid, point, idx, ts, extra):
        return Violation(typ, sev, ctb, tid, idx, ts, eng.camera_id, tuple(point), extra=extra)

    def review(ev):
        f_rev.write(json.dumps({k: v for k, v in ev.items() if k not in ("pontos", "trilha")}, default=str) + "\n")
    buf = RingBuffer(int(fps * 5))  # últimos 5 s
    pack = EvidencePackager(os.path.join(args.out, "evidence"), fps)
    queue, detran = ViolationQueue(), DetranSim()
    f_inf = open(os.path.join(args.out, "infracoes.jsonl"), "w")
    f_dec = open(os.path.join(args.out, "decisoes.jsonl"), "w")

    idx, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and idx >= args.max_frames):
            break
        ts = args.start_time + idx / fps
        light = args.light
        if light == "auto":
            roi = eng.scene.get("traffic_light_roi")
            light = light_state(frame, roi) if roi else "unknown"

        tracks = trk.update(det(frame))
        for t in tracks:
            t.state["blinker"] = blink.update(t.id, frame, t.box)
            t.state["blinker_reliable"] = blink.reliable
        blink.forget({t.id for t in trk.tracks})
        forced = rot.apply(tracks, idx) if rot else []
        if f_trk:
            for t in tracks:
                f_trk.write(f"{idx},{idx / fps:.2f},{t.id},{t.cls}," + ",".join(f"{v:.0f}" for v in t.box) + "\n")
        violations = eng.evaluate(tracks, idx, ts, light)

        # manobras pela trajetória: verde / vermelho (infração) / amarelo (revisão)
        roi = eng.scene.get("maneuver_roi")  # onde a regra de manobra vale (fora: curva, borda, estacionamento)
        for t in tracks:
            ev = mw.update(t, idx, t.state.get("blinker_reliable", blink.reliable))
            if not ev or (roi and not point_in_polygon(t.point, roi)):
                continue
            ev["roteiro"] = bool(t.state.get("roteiro"))
            ov.add_maneuver(ev)
            if ev["cor"] == VERMELHO and not (lanes_in_scene and ev["tipo"] == "troca_faixa"):
                violations.append(make(MANOBRA_INFRACAO[ev["tipo"]], "grave", "Art. 196", t.id, t.point, idx, ts,
                                       {"lado": ev["lado"], "motivo": ev["motivo"], "seta_do_roteiro": ev["roteiro"]}))
            elif ev["cor"] != VERMELHO:
                review(ev)
        # batida e batida com fuga (a batida pode vir do roteiro; a fuga é sempre decidida pelo sistema)
        col_events = [cw.force(a, b, p, idx) for a, b, p in forced]
        col_events += cw.update(tracks, {t.id for t in trk.tracks}, idx)
        for ev in col_events:
            if ev["tipo"] == "colisao":
                ov.add_collision(ev)
                review(ev)
            elif ev["tipo"] == "fuga":
                ov.add_fuga(ev)
                violations.append(make("evasao_apos_colisao", "gravissima", "Art. 176, I (conferir)",
                                       ev["track_id"], ev["ponto"], idx, ts,
                                       {"ficou": ev["ficou"], "batida_do_roteiro": ev.get("roteiro", False)}))
            else:
                review(ev)

        vis = frame.copy()
        draw_scene(vis, eng.scene)
        flagged = {v.track_id for v in violations}
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.box)
            color = (0, 0, 255) if t.state.get("reported") or t.id in flagged else (230, 230, 230)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
            cv2.putText(vis, f"{t.cls} #{t.id}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        ov.draw(vis, idx, tracks)
        cv2.putText(vis, f"t={idx / fps:5.1f}s  frame {idx}  semaforo: {light}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        buf.push(frame)

        for v in violations:
            queue.push_violation(v, v.severity)
            f_inf.write(json.dumps(v.to_dict(), default=str) + "\n")
        while len(queue):  # aqui envia tudo; em produção, só o que a banda permitir
            v = queue.pop()
            pack.package(v, buf.snapshot(), vis)
            f_dec.write(json.dumps(detran.receive(v.to_dict()), default=str) + "\n")
        if flagged:
            cv2.putText(vis, "INFRACAO", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        writer.write(vis)
        idx += 1

    writer.release()
    f_inf.close()
    f_dec.close()
    f_rev.close()
    if f_trk:
        f_trk.close()
    if rot:
        with open(os.path.join(args.out, "roteiro_aplicado.json"), "w", encoding="utf-8") as f:
            json.dump(rot.log, f, ensure_ascii=False, indent=1)
        nao = [e for e in rot.log if "batida_nao_achada" in e] + \
              [s for s in rot.seta if s["track"] is None]
        if nao:
            print("ATENÇÃO: anotações do roteiro que não acharam veículo:", nao)
    dt = time.time() - t0
    print(f"{idx} frames em {dt:.1f}s ({idx / max(dt, 1e-9):.1f} fps). Saída: {args.out}")


if __name__ == "__main__":
    main()
