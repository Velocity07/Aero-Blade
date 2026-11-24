import json
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_normalized_adjacency() -> np.ndarray:
    adj = np.eye(6, dtype=np.float32)
    for upwind, downwind in [(0, 3), (1, 4), (2, 5)]:
        adj[downwind, upwind] = 0.65
    for src, dst in [(0, 1), (1, 0), (1, 2), (2, 1), (3, 4), (4, 3), (4, 5), (5, 4)]:
        adj[dst, src] = 0.20
    return adj / adj.sum(axis=1, keepdims=True)


def run_farm_inference() -> list[dict]:
    print("=== Aero-Blade ONNX Graph & NLP Inference Engine ===\n")

    # 1. Load node features produced by Milestone 2 DSP worker
    df = pd.read_csv("data/dsp_node_features.csv")

    features = np.column_stack(
        [
            df["avg_wind_ms"].values / 15.0,
            df["avg_rpm"].values / 20.0,
            df["avg_bpf_hz"].values / 1.5,
            df["avg_hf_db"].values / 80.0,
            df["acoustic_anomaly_index"].values / 25.0,
        ]
    ).astype(np.float32)

    adj = build_normalized_adjacency()

    # 2. Run Spatial Graph Neural Network via ONNX Runtime
    gnn_session = ort.InferenceSession("models/farm_gnn.onnx")
    gnn_outputs = gnn_session.run(
        None,
        {"node_features": features, "adjacency_matrix": adj},
    )[0]

    # 3. Load Historical Maintenance Logs & NLP ONNX Encoder
    logs_df = pd.read_csv("data/maintenance_logs.csv")
    with open("models/tfidf_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vectorizer = TfidfVectorizer(vocabulary=vocab, stop_words="english")
    log_tfidf = vectorizer.fit_transform(
        logs_df["notes"]).toarray().astype(np.float32)

    nlp_session = ort.InferenceSession("models/nlp_log_matcher.onnx")
    # Encode each maintenance log one row at a time to match ONNX input shape [1, vocab_size]
    log_embeddings = np.vstack(
        [
            nlp_session.run(None, {"tfidf_features": log_tfidf[i: i + 1]})[0]
            for i in range(len(log_tfidf))
        ]
    )

    # 4. Evaluate each turbine and match the closest historical maintenance ticket
    diagnostics = []
    for idx, row in df.iterrows():
        pitting_prob = float(gnn_outputs[idx, 0])
        wake_prob = float(gnn_outputs[idx, 1])

        if pitting_prob > 0.50:
            status = "CRITICAL: BLADE MICRO-PITTING"
            query = "High-frequency acoustic whistle blade passing frequency leading-edge pitting"
        elif wake_prob > 0.45:
            status = "ADVISORY: DOWNWIND WAKE TURBULENCE"
            query = "Downwind turbulence wind speed deficit upwind rotor wake intact"
        else:
            status = "HEALTHY"
            query = "Normal broadband aerodynamic whoosh balanced root strain"

        query_vec = vectorizer.transform([query]).toarray().astype(np.float32)
        query_emb = nlp_session.run(None, {"tfidf_features": query_vec})[0]

        # Cosine similarity between live turbine state and historical technician logs
        similarities = (log_embeddings @ query_emb.T).squeeze()
        best_match_idx = int(np.argmax(similarities))
        matched_log = logs_df.iloc[best_match_idx]

        diagnostics.append(
            {
                "turbine_id": row["turbine_id"],
                "row": int(row["row"]),
                "pitting_prob": round(pitting_prob, 3),
                "wake_impact_prob": round(wake_prob, 3),
                "status": status,
                "matched_log_id": matched_log["log_id"],
                "recommended_action": matched_log["recommended_action"],
            }
        )

    results_df = pd.DataFrame(diagnostics)
    results_df.to_csv("data/farm_diagnostics.csv", index=False)
    print(results_df.to_string(index=False))
    return diagnostics


if __name__ == "__main__":
    run_farm_inference()
