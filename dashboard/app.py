"""Dashboard: streamlit run dashboard/app.py -- --run runs/demo"""
import glob
import json
import os
import sys

import pandas as pd
import streamlit as st

run = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else "runs/demo"
st.set_page_config(page_title="CompQuest — Fiscalização por câmera", layout="wide")
st.title("Fiscalização de trânsito por câmera")
st.caption(f"Execução: {run}")


def load(name):
    p = os.path.join(run, name)
    if not os.path.exists(p):
        return pd.DataFrame()
    with open(p) as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


inf, dec = load("infracoes.jsonl"), load("decisoes.jsonl")
c1, c2, c3 = st.columns(3)
c1.metric("Infrações detectadas", len(inf))
c2.metric("Multas aplicadas", int((dec.get("multa", pd.Series(dtype=float)) > 0).sum()) if len(dec) else 0)
c3.metric("Advertências", int((dec.get("acao", pd.Series(dtype=str)) == "advertencia").sum()) if len(dec) else 0)

video = os.path.join(run, "video_anotado.mp4")
if os.path.exists(video):
    st.video(video)

if len(inf):
    st.subheader("Infrações por tipo")
    st.bar_chart(inf["type"].value_counts())
    st.subheader("Registro")
    st.dataframe(inf[["frame_idx", "type", "severity", "ctb", "track_id", "points"]], use_container_width=True)
if len(dec):
    st.subheader("Decisões do Detran simulado")
    st.dataframe(dec, use_container_width=True)

st.subheader("Evidências")
for folder in sorted(glob.glob(os.path.join(run, "evidence", "*")))[:20]:
    with st.expander(os.path.basename(folder)):
        img = os.path.join(folder, "frame.jpg")
        if os.path.exists(img):
            st.image(img)
        st.json(json.load(open(os.path.join(folder, "evidence.json"))))
