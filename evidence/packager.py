"""Monta o pacote de evidência: clipe (do ring buffer), frame-chave anotado, JSON e hash.

Cadeia de custódia: cada pacote guarda o SHA-256 do anterior (lista encadeada de hashes).
Se alguém alterar um pacote antigo, a cadeia quebra.
"""
import hashlib
import json
import os
from typing import List

import cv2


class EvidencePackager:
    def __init__(self, out_dir: str = "evidence/out", fps: float = 30.0):
        self.out_dir = out_dir
        self.fps = fps
        os.makedirs(out_dir, exist_ok=True)
        self.prev_hash = "0" * 64

    def package(self, violation, frames: List, key_frame) -> str:
        vid = f"{violation.camera_id}_{violation.frame_idx}_{violation.track_id}_{violation.type}"
        folder = os.path.join(self.out_dir, vid)
        os.makedirs(folder, exist_ok=True)

        clip_path = os.path.join(folder, "clip.mp4")
        if frames:
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            for f in frames:
                vw.write(f)
            vw.release()
        cv2.imwrite(os.path.join(folder, "frame.jpg"), key_frame)

        meta = violation.to_dict()
        meta["files"] = ["clip.mp4", "frame.jpg"]
        digest = hashlib.sha256()
        for name in meta["files"]:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    digest.update(fh.read())
        digest.update(json.dumps(meta, sort_keys=True, default=str).encode())
        digest.update(self.prev_hash.encode())
        meta["prev_hash"] = self.prev_hash
        meta["hash"] = digest.hexdigest()
        self.prev_hash = meta["hash"]

        with open(os.path.join(folder, "evidence.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        return folder
