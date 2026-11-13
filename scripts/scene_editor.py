"""Editor de cena: clica no 1º frame do vídeo para marcar linhas e polígonos.

python scripts/scene_editor.py --video data/videos/minha_camera.mp4 --out data/scenes/minha_camera.json
python scripts/scene_editor.py --image data/raw/wibam_demo/frames/0/29732.jpg --out data/scenes/wibam_c0.json

Teclas:
  s = linha de retenção (2 cliques)     l = linha contínua (2 cliques)
  d = linha tracejada entre faixas (2 cliques; usada na troca de faixa sem seta)
  c = faixa de pedestre (polígono)      b = faixa de ônibus (polígono)       g = área proibida p/ "gato" (canteiro, ilha, calçada, terra)
  z = zona/faixa (polígono, depois pergunta id e sentido no terminal)
  t = ROI do semáforo (2 cliques: canto sup. esq. e inf. dir.)
  ENTER = fecha o polígono atual    w = salva JSON    q = sai
Depois edite "allowed_moves" no JSON: pares [zona_origem, zona_destino] permitidos.
"""
import argparse
import json

import cv2
import numpy as np

state = {"mode": None, "pts": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--image", help="alternativa a --video: um frame .jpg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera-id", default="cam0")
    args = ap.parse_args()

    assert args.video or args.image, "passe --video ou --image"
    if args.image:
        frame = cv2.imread(args.image)
        ok = frame is not None
    else:
        ok, frame = cv2.VideoCapture(args.video).read()
    assert ok, "não consegui ler o vídeo/imagem"
    h, w = frame.shape[:2]
    scene = {"camera_id": args.camera_id, "frame_size": [w, h], "stop_line": None, "traffic_light_roi": None,
             "crosswalk": None, "solid_lines": [], "lane_lines": [], "lanes": [], "bus_lane": None, "forbidden_areas": [], "zones": [], "allowed_moves": []}

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and state["mode"]:
            state["pts"].append([x, y])
            m, p = state["mode"], state["pts"]
            if m in ("s", "l", "d", "t") and len(p) == 2:
                if m == "s":
                    scene["stop_line"] = p
                elif m == "l":
                    scene["solid_lines"].append(p)
                elif m == "d":
                    scene["lane_lines"].append(p)
                else:
                    scene["traffic_light_roi"] = [p[0][0], p[0][1], p[1][0], p[1][1]]
                state.update(mode=None, pts=[])

    cv2.namedWindow("cena")
    cv2.setMouseCallback("cena", on_click)
    while True:
        vis = frame.copy()
        for z in scene["zones"]:
            cv2.polylines(vis, [np.array(z["polygon"])], True, (255, 200, 0), 2)
        if scene["stop_line"]:
            cv2.line(vis, *map(tuple, scene["stop_line"]), (0, 0, 255), 2)
        for a, b in scene["solid_lines"]:
            cv2.line(vis, tuple(a), tuple(b), (0, 255, 255), 2)
        for a, b in scene["lane_lines"]:
            cv2.line(vis, tuple(a), tuple(b), (200, 200, 200), 1)
        for fa in scene["forbidden_areas"]:
            cv2.polylines(vis, [np.array(fa["polygon"])], True, (0, 0, 255), 2)
        if scene["crosswalk"]:
            cv2.polylines(vis, [np.array(scene["crosswalk"])], True, (255, 255, 255), 2)
        for p in state["pts"]:
            cv2.circle(vis, tuple(p), 4, (0, 255, 0), -1)
        cv2.putText(vis, f"modo: {state['mode']}  (s l d c b z t g | ENTER w q)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("cena", vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k in map(ord, "sldcbztg"):
            state.update(mode=chr(k), pts=[])
        if k == 13 and state["mode"] in ("c", "b", "z", "g") and len(state["pts"]) >= 3:
            poly = state["pts"]
            if state["mode"] == "c":
                scene["crosswalk"] = poly
            elif state["mode"] == "g":
                tipo = input("tipo (canteiro/ilha/calcada/terra): ").strip() or "canteiro"
                scene["forbidden_areas"].append({"tipo": tipo, "polygon": poly})
            elif state["mode"] == "b":
                scene["bus_lane"] = {"polygon": poly, "hours": [[6, 9], [17, 20]]}
            else:
                zid = input("id da zona: ").strip()
                d = input("sentido do trânsito na imagem (ex.: 0,1 = descendo; 1,0 = para direita): ")
                one = input("mão única? (s/n): ").strip().lower() == "s"
                if input("é faixa de rolamento paralela (vale p/ troca de faixa)? (s/n): ").strip().lower() == "s":
                    scene["lanes"].append(zid)
                scene["zones"].append({"id": zid, "polygon": poly,
                                       "direction": [float(v) for v in d.split(",")], "one_way": one})
            state.update(mode=None, pts=[])
        if k == ord("w"):
            with open(args.out, "w") as f:
                json.dump(scene, f, indent=2)
            print("salvo em", args.out)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
