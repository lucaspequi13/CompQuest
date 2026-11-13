"""Editor do modo roteiro: assiste o vídeo, clica no veículo e marca seta / batida.

python scripts/roteiro_editor.py --video data/videos/xxx.mp4 [--out data/roteiros/xxx.json] [--dur 3]

Teclas (com a janela do vídeo em foco):
  espaço  pausa / continua            j / l  volta / avança 1 s       , / .  volta / avança 1 frame
  clique  marca o veículo (ponto) no frame atual
  1       seta ESQUERDA a partir de agora (por --dur s)     2  seta DIREITA
  0       conferido: SEM seta (a partir de agora, por --dur s)
  b       BATIDA: clique no 1º veículo e b; clique no 2º veículo e b de novo (b duas vezes no mesmo
          clique = usa os 2 veículos mais perto do ponto)
  u       desfaz a última marcação     s  salva     q / Esc  salva e sai
Lado da seta = do ponto de vista do MOTORISTA.
Depois: python scripts/run_pipeline.py --video ... --scene ... --roteiro <arquivo>.json
"""
import argparse
import json
import os

import cv2

KEYS = {ord("1"): "esquerda", ord("2"): "direita", ord("0"): "nenhuma"}
COLORS = {"esquerda": (60, 200, 60), "direita": (60, 200, 60), "nenhuma": (40, 40, 235)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out")
    ap.add_argument("--dur", type=float, default=3.0, help="duração padrão da seta marcada (s)")
    ap.add_argument("--width", type=int, default=1280, help="largura da janela (o vídeo é reduzido se maior)")
    args = ap.parse_args()
    out = args.out or os.path.join("data", "roteiros", os.path.splitext(os.path.basename(args.video))[0] + ".json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    data = {"video": args.video.replace("\\", "/"), "seta": [], "batida": []}
    if os.path.exists(out):
        with open(out, encoding="utf-8-sig") as f:
            data.update(json.load(f))
        print(f"Continuando {out}: {len(data['seta'])} seta(s), {len(data['batida'])} batida(s)")
    history = []

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4))
    scale = min(1.0, args.width / W)
    state = {"click": None, "idx": 0, "paused": True}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (round(x / scale), round(y / scale))

    win = "roteiro (espaco=play, clique=veiculo, 1/2/0=seta, b=batida, s=salvar, q=sair)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    def save():
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"salvo: {out}")

    frame = None
    while True:
        if frame is None or not state["paused"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, state["idx"])
            ok, frame = cap.read()
            if not ok:
                state["paused"], state["idx"] = True, max(0, n - 1)
                continue
        t = state["idx"] / fps
        vis = frame.copy()
        for s in data["seta"]:
            if s["t0"] <= t <= s["t1"]:
                cv2.circle(vis, (int(s["x"]), int(s["y"])), 10, COLORS[s["lado"]], 3)
                cv2.putText(vis, f"seta {s['lado']}", (int(s["x"]) + 12, int(s["y"])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[s["lado"]], 2)
        for b in data["batida"]:
            if abs(b["t"] - t) < 1.0:
                cv2.drawMarker(vis, (int(b["x"]), int(b["y"])), (40, 40, 235), cv2.MARKER_TILTED_CROSS, 30, 3)
        if state["click"]:
            cv2.drawMarker(vis, state["click"], (0, 215, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(vis, f"t={t:6.2f}s  frame {state['idx']}/{n}  {'PAUSADO' if state['paused'] else ''}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if scale < 1.0:
            vis = cv2.resize(vis, (int(W * scale), int(H * scale)))
        cv2.imshow(win, vis)

        k = cv2.waitKey(int(1000 / fps) if not state["paused"] else 30) & 0xFF
        if not state["paused"]:
            state["idx"] = min(n - 1, state["idx"] + 1)
        if k == 255:
            continue
        if k == ord(" "):
            state["paused"] = not state["paused"]
        elif k in (ord("j"), ord("l"), ord(","), ord(".")):
            step = {ord("j"): -fps, ord("l"): fps, ord(","): -1, ord("."): 1}[k]
            state["idx"] = int(max(0, min(n - 1, state["idx"] + step)))
            frame = None
        elif k in KEYS or k == ord("b"):
            if not state["click"]:
                print("clique no veículo primeiro")
                continue
            x, y = state["click"]
            if k == ord("b"):
                pend = state.get("batida")
                if pend is None:
                    state["batida"] = {"t": round(t, 2), "x": x, "y": y}
                    print("batida: 1º veículo marcado; clique no 2º e aperte b")
                    continue
                item, key = dict(pend), "batida"
                if (x, y) != (pend["x"], pend["y"]):
                    item.update(x2=x, y2=y)
                state["batida"] = None
            else:
                item, key = {"t0": round(t, 2), "t1": round(t + args.dur, 2), "x": x, "y": y, "lado": KEYS[k]}, "seta"
            data[key].append(item)
            history.append(key)
            print(key, item)
        elif k == ord("u") and history:
            print("desfeito:", data[history[-1]].pop())
            history.pop()
        elif k == ord("s"):
            save()
        elif k in (ord("q"), 27):
            save()
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
