"""Pipeline ponta a ponta sobre um vídeo.

python scripts/run_pipeline.py --video data/videos/car-detection.mp4 \
    --scene data/scenes/exemplo_car_detection.json --out runs/demo

Saídas em --out: video_anotado.mp4, infracoes.jsonl, decisoes.jsonl e evidence/
"""
import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detran_sim.scoring import DetranSim  # noqa: E402
from eda.priority_heap import ViolationQueue  # noqa: E402
from eda.ring_buffer import RingBuffer  # noqa: E402
from evidence.packager import EvidencePackager  # noqa: E402
from perception.blinker import BlinkerClassifier  # noqa: E402
from perception.detector import Detector  # noqa: E402
from perception.tracker import Tracker  # noqa: E402
from perception.traffic_light import light_state  # noqa: E402
from rules.engine import RuleEngine  # noqa: E402


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
    ap.add_argument("--start-time", type=float, default=time.time(), help="epoch do 1º frame")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(os.path.join(args.out, "video_anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    det, trk, eng = Detector(args.weights), Tracker(), RuleEngine(args.scene)
    blink = BlinkerClassifier(fps, args.blinker_model)
    print("Seta:", "RandomForest" if blink.model else "baseline heurístico (sem modelo treinado)")
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
        blink.forget({t.id for t in trk.tracks})
        violations = eng.evaluate(tracks, idx, ts, light)

        vis = frame.copy()
        draw_scene(vis, eng.scene)
        flagged = {v.track_id for v in violations}
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.box)
            color = (0, 0, 255) if t.state.get("reported") else (0, 200, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            seta = t.state.get("blinker", "nenhuma")
            label = f"{t.cls} #{t.id}" + ("" if seta == "nenhuma" else f" seta:{seta}")
            cv2.putText(vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(vis, f"frame {idx}  semaforo: {light}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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
    dt = time.time() - t0
    print(f"{idx} frames em {dt:.1f}s ({idx / max(dt, 1e-9):.1f} fps). Saída: {args.out}")


if __name__ == "__main__":
    main()
