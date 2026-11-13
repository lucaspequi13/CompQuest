"""API do Detran simulado. Rodar: uvicorn detran_sim.api:app --reload"""
from fastapi import FastAPI

from detran_sim.scoring import DetranSim

app = FastAPI(title="Detran simulado — CompQuest")
sim = DetranSim()


@app.post("/infracoes")
def receber(v: dict):
    return sim.receive(v)


@app.get("/condutores/{placa}")
def condutor(placa: str):
    d = sim.drivers.get(placa)
    if not d:
        return {"placa": placa, "eventos": []}
    return {"placa": placa, "eventos": [e for _, e in d.all_events.range(0, float("inf"))]}


@app.get("/log")
def log():
    return sim.log[-200:]
