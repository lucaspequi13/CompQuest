"""Pipeline ponta a ponta sobre um vídeo.

python scripts/run_pipeline.py --video data/videos/car-detection.mp4 \
    --scene data/scenes/exemplo_car_detection.json --out runs/demo

Saídas em --out: video_anotado.mp4, infracoes.jsonl, decisoes.jsonl, revisao.jsonl e evidence/
Vídeo: só veículos (--all-classes para ver tudo); rastro verde/vermelho/amarelo nas manobras; BATIDA / FUGA.
"""
import argparse
import heapq
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
from eda.plate_trie import is_valid_plate  # noqa: E402
from perception.detector import VEHICLES, Detector  # noqa: E402
from perception.plate_reader import PlateRegistry  # noqa: E402
from perception.tracker import Tracker  # noqa: E402
from perception.traffic_light import light_state  # noqa: E402
from rules.collision import CollisionWatcher  # noqa: E402
from rules.engine import RuleEngine  # noqa: E402
from rules.maneuver import VERMELHO, ManeuverWatcher, maneuver_from_trail  # noqa: E402
from rules.roteiro import Roteiro  # noqa: E402
from rules.types import Violation  # noqa: E402

MANOBRA_INFRACAO = {"troca_faixa": "troca_faixa_sem_seta", "conversao": "conversao_sem_seta"}
# texto curto no vídeo para as infrações das regras (rules/violations/checks.py)
ROTULO = {"avanco_sinal_vermelho": "AVANCOU O VERMELHO", "contramao": "CONTRAMAO",
          "transpor_linha_continua": "CRUZOU LINHA CONTINUA", "conversao_proibida": "CONVERSAO PROIBIDA",
          "parada_sobre_faixa_pedestre": "PAROU NA FAIXA", "faixa_exclusiva_onibus": "FAIXA DE ONIBUS",
          "gato_canteiro": "GATO: PASSOU NO CANTEIRO", "gato_marcha_re": "GATO: MARCHA A RE",
          "troca_faixa_sem_seta": "TROCOU DE FAIXA SEM SETA"}


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
    ap.add_argument("--conf", type=float, default=0.35, help="confiança mínima do YOLO (baixar mantém veículo capotado/borrado)")
    ap.add_argument("--min-box", type=float, default=0.06,
                    help="altura mínima da caixa (fração da imagem) p/ manobra e batida. Vista de cima/drone: 0.02")
    ap.add_argument("--plate-weights", default="oim",
                    help="'oim' (detector de placa pronto, open-image-models), caminho de YOLO 1 classe 'plate', "
                         "ou 'none' (só a região padrão da caixa)")
    ap.add_argument("--ocr", default="fast", choices=["fast", "easy", "none"],
                    help="OCR da placa: fast-plate-ocr (padrão), easyocr ou none")
    ap.add_argument("--frames-out", help="pasta para salvar as FOTOS dos momentos com mais acontecimentos (PNG sem perda)")
    ap.add_argument("--frames-n", type=int, default=8, help="quantas fotos salvar")
    ap.add_argument("--frames-gap", type=float, default=1.0, help="distância mínima entre fotos (s)")
    ap.add_argument("--dump-tracks", action="store_true", help="grava tracks.csv (frame, id, classe, caixa)")
    ap.add_argument("--all-classes", action="store_true", help="mostrar também pedestre/bicicleta")
    ap.add_argument("--start-time", type=float, default=time.time(), help="epoch do 1º frame")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    writer = cv2.VideoWriter(os.path.join(args.out, "video_anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    det = Detector(args.weights, conf=args.conf, keep=None if args.all_classes else VEHICLES)
    trk, eng = Tracker(), RuleEngine(args.scene)
    blink = BlinkerClassifier(fps, args.blinker_model)
    print("Seta:", "RandomForest" if blink.model else "baseline heurístico (sem modelo treinado)",
          "| acusa sem seta:", "sim" if blink.reliable else "não (manobras vão para revisão, amarelo)")
    plates = PlateRegistry(None if args.plate_weights == "none" else args.plate_weights,
                           None if args.ocr == "none" else args.ocr)
    print("Placa:", "detector" if plates.detector else "região padrão da caixa", "| OCR:", plates.ocr_name or "nenhum",
          "(sem OCR: guarda o melhor recorte e marca ILEGIVEL)")
    painel = []  # infratores na lista de placas: dicts {tid, crop, txt, cls, rot}
    cls_of = {}  # track id -> classe (para "car #8" mesmo depois que o veículo sai de cena)
    mw = ManeuverWatcher(fps, min_box_px=args.min_box * h, frame_size=(w, h),
                         roi=RuleEngine(args.scene).scene.get("maneuver_roi"))
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

    def monitor(tid, rot):
        """Põe o veículo na lista de placas monitoradas no PRIMEIRO evento (batida, manobra, infração)."""
        if tid is None or not isinstance(tid, int) or any(p["tid"] == tid for p in painel):
            return
        crop = plates.best(tid)
        painel.insert(0, {"tid": tid, "crop": None if crop is None else crop.copy(), "txt": plates.text(tid),
                          "cls": cls_of.get(tid, "veic"), "rot": rot})

    def review(ev):
        f_rev.write(json.dumps({k: v for k, v in ev.items() if k not in ("pontos", "trilha")}, default=str) + "\n")
    buf = RingBuffer(int(fps * 5))  # últimos 5 s
    pack = EvidencePackager(os.path.join(args.out, "evidence"), fps)
    queue, detran = ViolationQueue(), DetranSim()
    f_inf = open(os.path.join(args.out, "infracoes.jsonl"), "w")
    f_dec = open(os.path.join(args.out, "decisoes.jsonl"), "w")

    # fotos dos momentos com mais acontecimentos: o melhor frame de cada janela de --frames-gap segundos
    # (PNG codificado só quando bate o recorde da janela); no fim, heap (EDA) escolhe as N janelas de maior nota
    best_win, win = {}, max(1, int(args.frames_gap * fps))

    def intensidade(new_violations, observing, light_now):
        kinds = [k for k, _ in ov.active]
        nomes = {"manobra": "manobra", "colisao": "batida", "fuga": "fuga", "alerta": "infracao",
                 "ficaram": "ficaram"}
        score = 3.0 * len(new_violations) + 2.0 * sum(k in ("colisao", "fuga", "alerta") for k in kinds)
        score += 1.5 * sum(k == "manobra" and ev.get("cor") == VERMELHO for k, ev in ov.active)
        score += 0.7 * sum(k == "manobra" for k in kinds) + 1.0 * len(observing) + 0.5 * min(3, len(painel))
        score += 0.5 if light_now in ("red", "yellow") else 0.0
        tags = sorted({nomes[k] for k in kinds if k in nomes} | ({"placa"} if painel else set())
                      | ({"sinal-" + light_now} if light_now in ("red", "yellow") else set()))
        return score, "+".join(tags) or "transito"

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
        cls_of.update({t.id: t.cls for t in tracks})
        # sempre registrando o melhor recorte da placa de cada veículo (ignorando logos/botões gravados no vídeo)
        plates.update(frame, tracks, idx, eng.scene.get("plate_ignore", []))
        for t in tracks:
            t.state["blinker"] = blink.update(t.id, frame, t.box)
            t.state["blinker_reliable"] = blink.reliable
        blink.forget({t.id for t in trk.tracks})
        forced = rot.apply(tracks, idx) if rot else []
        if f_trk:
            for t in tracks:
                f_trk.write(f"{idx},{idx / fps:.2f},{t.id},{t.cls}," + ",".join(f"{v:.0f}" for v in t.box) + "\n")
        violations = eng.evaluate(tracks, idx, ts, light)
        by_id = {t.id: t for t in tracks}
        for v in violations:
            if v.track_id in by_id:
                ov.add_alert(idx, by_id[v.track_id], ROTULO.get(v.type, v.type.upper()))

        # manobras pela trajetória: verde / vermelho (infração) / amarelo (revisão)
        roi = eng.scene.get("maneuver_roi")  # onde a regra de manobra vale (fora: curva, borda, estacionamento)
        rel = lambda tr: tr.state.get("blinker_reliable", blink.reliable)  # noqa: E731
        evs = [(t, mw.update(t, idx, rel(t))) for t in tracks]
        evs += [(ev.pop("_track"), ev) for ev in mw.flush({t.id for t in trk.tracks}, idx, rel)]
        for t, ev in evs:
            if not ev:
                continue
            ev.pop("_track", None)
            # região de fiscalização: vale onde a manobra FOI DETECTADA (gatilho). A trajetória inteira pode
            # vir de fora (carro que sai da rua lateral) e, quando a manobra é decidida, ele já pode ter saído.
            if roi and not point_in_polygon(ev.get("gatilho", ev["pontos"][-1]), roi):
                continue
            if eng.scene.get("maneuver_tipo"):  # calibração da câmera: num cruzamento, manobra = conversão
                ev["tipo"] = eng.scene["maneuver_tipo"]
            ev["roteiro"] = bool(t.state.get("roteiro"))
            ov.add_maneuver(ev)
            monitor(t.id, "MANOBRA")
            if ev["cor"] == VERMELHO and not (lanes_in_scene and ev["tipo"] == "troca_faixa"):
                violations.append(make(MANOBRA_INFRACAO[ev["tipo"]], "grave", "Art. 196", t.id, t.point, idx, ts,
                                       {"lado": ev["lado"], "motivo": ev["motivo"], "seta_do_roteiro": ev["roteiro"]}))
            elif ev["cor"] != VERMELHO:
                review(ev)
        # manobra marcada no roteiro (quando a geometria não confirma sozinha): rota real, lado pela geometria
        for t, n, tipo in (rot.maneuvers(tracks, idx) if rot else []):
            ev = maneuver_from_trail(t, n, tipo, idx, t.state.get("blinker_reliable", blink.reliable))
            ov.add_maneuver(ev)
            monitor(t.id, "MANOBRA")
            if ev["cor"] == VERMELHO:
                violations.append(make(MANOBRA_INFRACAO.get(tipo, "troca_faixa_sem_seta"), "grave", "Art. 196",
                                       t.id, t.point, idx, ts, {"lado": ev["lado"], "motivo": ev["motivo"],
                                                                "seta_do_roteiro": True, "manobra_do_roteiro": True}))
            else:
                review(ev)
        # batida e batida com fuga (a batida pode vir do roteiro; a fuga é sempre decidida pelo sistema)
        col_events = [cw.force(a, b, p, idx, leaver) for a, b, p, leaver in forced]
        col_events += cw.update(tracks, {t.id for t in trk.tracks}, idx)
        for ev in col_events:
            if ev["tipo"] == "colisao":
                ov.add_collision(ev)
                review(ev)
                for tid in ev["ids"]:
                    monitor(tid, "BATIDA")
            elif ev["tipo"] == "ficaram":
                ov.add_ficaram(ev)
                review(ev)
            elif ev["tipo"] == "fuga":
                ov.add_fuga(ev)
                violations.append(make("evasao_apos_colisao", "gravissima", "Art. 176, I (conferir)",
                                       ev["track_id"], ev["ponto"], idx, ts,
                                       {"ficou": [str(x) for x in ev["ficou"]],
                                        "batida_do_roteiro": ev.get("roteiro", False),
                                        "fuga_do_roteiro": ev.get("fuga_do_roteiro", False)}))
            else:
                review(ev)

        # placa de cada infrator: texto (ou ILEGIVEL) + melhor recorte -> Detran, evidência e painel
        for v in violations:
            v.plate = plates.text(v.track_id)
            v.extra["placa_status"] = ("ilegivel: recorte na evidencia" if v.plate == "ILEGIVEL" else
                                       "lida (padrao BR)" if is_valid_plate(v.plate) else "lida (outro pais)")
            monitor(v.track_id, ROTULO.get(v.type, v.type.replace("_", " ").upper()))
        for p in painel:  # enquanto o veículo está em cena, o recorte/leitura pode melhorar
            if plates.best(p["tid"]) is not None:
                p["crop"], p["txt"] = plates.best(p["tid"]), plates.text(p["tid"])
        plates.forget({t.id for t in trk.tracks}, keep={p["tid"] for p in painel[:3]})

        vis = frame.copy()
        draw_scene(vis, eng.scene)
        flagged = {v.track_id for v in violations}
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.box)
            ev_color = ov.track_color(t.id, idx)  # caixa na cor da rota/evento do veículo
            if ev_color is None and t.id in flagged:
                ev_color = (0, 0, 255)
            color, thick = (ev_color, 2) if ev_color else ((230, 230, 230), 1)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
            cv2.putText(vis, f"{t.cls} #{t.id}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45 if ev_color else 0.4,
                        color, thick)
            if y2 - y1 >= 40:  # quadradinho da placa: o sistema sabe onde ela está (mesmo borrada/ilegível)
                ov.draw_plate_box(vis, plates.loc.get(t.id), ev_color or (255, 255, 0), 2 if ev_color else 1)
        ov.draw(vis, idx, tracks)
        ov.draw_light(vis, eng.scene.get("traffic_light_roi"), light)
        ov.draw_observing(vis, cw.observing(idx), n_frames - 1 - idx, fps)
        # lista de placas: infratores primeiro, depois os demais envolvidos (mais recentes primeiro)
        ordem = sorted(painel, key=lambda p: p["rot"] in ("MANOBRA", "BATIDA"))
        ov.draw_plate_panel(vis, [(f"{p['cls']} #{p['tid']}", p["txt"]) for p in ordem])
        cv2.putText(vis, f"t={idx / fps:5.1f}s  frame {idx}  semaforo: {light}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        buf.push(frame)

        for v in violations:
            queue.push_violation(v, v.severity)
            f_inf.write(json.dumps(v.to_dict(), default=str) + "\n")
        while len(queue):  # aqui envia tudo; em produção, só o que a banda permitir
            v = queue.pop()
            crop = plates.best(v.track_id)
            if crop is None:
                crop = next((p["crop"] for p in painel if p["tid"] == v.track_id), None)
            pack.package(v, buf.snapshot(), vis, plate_crop=crop)
            f_dec.write(json.dumps(detran.receive(v.to_dict()), default=str) + "\n")
        if flagged:
            cv2.putText(vis, "INFRACAO", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        writer.write(vis)
        if args.frames_out:
            sc, tags = intensidade(violations, cw.observing(idx), light)
            w_id = idx // win
            if sc > 0 and sc > best_win.get(w_id, (0,))[0]:
                ok_png, png = cv2.imencode(".png", vis)
                best_win[w_id] = (sc, idx, tags, png.tobytes())
        idx += 1

    writer.release()
    if args.frames_out and best_win:
        os.makedirs(args.frames_out, exist_ok=True)
        picked, min_gap = [], int(0.8 * win)  # janelas vizinhas podem ter fotos coladas: exige distância
        for c in heapq.nlargest(len(best_win), best_win.values(), key=lambda c: (c[0], -c[1])):
            if all(abs(c[1] - p[1]) >= min_gap for p in picked):
                picked.append(c)
            if len(picked) >= args.frames_n:
                break
        chosen = sorted(picked, key=lambda c: c[1])  # em ordem de tempo: as fotos contam a sequência
        for rank, (sc, fi, tags, png) in enumerate(chosen, 1):
            name = f"{rank:02d}_t{fi / fps:05.1f}s_{tags}.png".replace("+", "_")
            with open(os.path.join(args.frames_out, name), "wb") as fh:
                fh.write(png)
        print(f"{len(chosen)} fotos em {args.frames_out}")
    for ev in cw.finish(idx):  # batidas que o vídeo não deixou concluir
        review(ev)
        print(f"Batida {ev['ids']} inconclusiva: {ev['motivo']}")
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
