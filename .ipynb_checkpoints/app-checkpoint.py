import gradio as gr
import pandas as pd
import numpy as np
import joblib

bundle = joblib.load("energi_model.joblib")
model = bundle["model"]
TRAIN_COLS = bundle["columns"]

MATERIAL_KOLONNER = ["mat_Betong", "mat_Murteglstein", "mat_Stål", "mat_Tre", "mat_Ukjent"]
MATERIAL_MAP = {
    "Betong": "mat_Betong",
    "Mur/teglstein": "mat_Murteglstein",
    "Stål": "mat_Stål",
    "Tre": "mat_Tre",
    "Ukjent": "mat_Ukjent",
}

def build_row(bygningskategori, byggear, postnummer, beregnetfossilandel, materialvalg):
    row = {c: 0 for c in TRAIN_COLS}

    row["bygningskategori"] = 0 if bygningskategori == "Småhus" else 1

    row["byggear"]      = int(byggear)
    row["postnummer"]   = int(postnummer)
    row["beregnetfossilandel"] = float(beregnetfossilandel)

    if materialvalg in MATERIAL_MAP:
        col = MATERIAL_MAP[materialvalg]
        if col in row:
            row[col] = 1

    X = pd.DataFrame([{c: row.get(c, 0) for c in TRAIN_COLS}])
    return X

def predict_fn(bygningskategori, byggear, postnummer, beregnetfossilandel, materialvalg, vis_sannsynligheter):
    X = build_row(bygningskategori, byggear, postnummer, beregnetfossilandel, materialvalg)

    pred = model.predict(X)[0]

    if vis_sannsynligheter and hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        return {cls: float(p) for cls, p in zip(model.classes_, probs)}
    else:
        return str(pred)

demo = gr.Interface(
    fn=predict_fn,
    inputs=[
        gr.Radio(["Småhus", "Boligblokker"], label="Bygningskategori", value="Småhus"),
        gr.Number(label="Byggeår", value=1990, precision=0),
        gr.Number(label="Postnummer", value=5000, precision=0),
        gr.Slider(0.0, 1.0, value=0.6, step=0.01, label="Beregnet fossilandel"),
        gr.Dropdown(["Tre", "Betong", "Mur/teglstein", "Stål", "Ukjent"], label="Materialvalg", value="Tre"),
        gr.Checkbox(label="Vis sannsynligheter", value=True),
    ],
    outputs=gr.Label(num_top_classes=7, label="Predikert energikarakter"),
    title="Energikarakter-modell",
    description="Demo av RandomForest-modell for energikarakter.",
)

if __name__ == "__main__":
    demo.queue()    
    demo.launch()   