import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


def build_wind_farm_adjacency() -> torch.Tensor:
    """
    Builds a normalized 6x6 directed graph adjacency matrix for the wind farm:
    Nodes 0,1,2 = T01,T02,T03 (Front Row)
    Nodes 3,4,5 = T04,T05,T06 (Back Row - Downwind)
    """
    adj = np.eye(6, dtype=np.float32)

    # Directed wake edges: Front row wind wake flows into Back row turbines
    wake_edges = [(0, 3), (1, 4), (2, 5)]
    for upwind, downwind in wake_edges:
        adj[downwind, upwind] = 0.65

    # Weak lateral neighbor awareness within the same row
    lateral_edges = [(0, 1), (1, 0), (1, 2), (2, 1),
                     (3, 4), (4, 3), (4, 5), (5, 4)]
    for src, dst in lateral_edges:
        adj[dst, src] = 0.20

    row_sums = adj.sum(axis=1, keepdims=True)
    return torch.tensor(adj / row_sums, dtype=torch.float32)


class FarmGraphNet(nn.Module):
    """
    GraphSAGE-style Spatial GNN: concatenates [own_features, neighbor_wake_context]
    so downwind turbines detect upwind wake turbulence without false-alarming on blade pitting.
    """

    def __init__(self, in_features: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neighbor_context = torch.matmul(adj, x)
        combined = torch.cat([x, neighbor_context], dim=-1)
        h = torch.relu(self.fc1(combined))
        return torch.sigmoid(self.fc2(h))


class NLPLogEncoder(nn.Module):
    """
    Trained neural text encoder projecting TF-IDF vectors into a
    16-dimensional unit-normalized semantic embedding space.
    """

    def __init__(self, vocab_size: int = 64, embed_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, tfidf_vec: torch.Tensor) -> torch.Tensor:
        emb = self.net(tfidf_vec)
        return emb / (torch.norm(emb, dim=-1, keepdim=True) + 1e-8)


def train_and_export() -> None:
    print("=== Training Precision Aero-Blade GNN & NLP Matcher ===\n")
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # 1. Generate 400 synthetic wind-farm graph snapshots for training
    adj = build_wind_farm_adjacency()
    X_list, Y_list = [], []

    for _ in range(400):
        base_wind = rng.uniform(8.0, 14.0)
        erosion = rng.uniform(0.02, 0.15, size=6)

        damaged_idx = int(rng.integers(0, 6)) if rng.random() < 0.8 else -1
        if damaged_idx != -1:
            erosion[damaged_idx] = rng.uniform(0.55, 0.95)

        feats = np.zeros((6, 5), dtype=np.float32)
        labels = np.zeros((6, 2), dtype=np.float32)

        for i in range(6):
            is_row2 = i >= 3
            upwind_damaged = is_row2 and (damaged_idx == i - 3)
            local_wind = base_wind * (0.82 if is_row2 else 1.0)
            rpm = local_wind * 1.35
            bpf = 3.0 * (rpm / 60.0)
            hf_db = 42.0 + (erosion[i] * 28.0) + \
                (2.5 if upwind_damaged else 0.0)
            anomaly_idx = (hf_db - 42.0) / ((310.0 + local_wind * 8.5) / 400.0)

            feats[i] = [
                local_wind / 15.0,
                rpm / 20.0,
                bpf / 1.5,
                hf_db / 80.0,
                anomaly_idx / 25.0,
            ]
            labels[i, 0] = 1.0 if erosion[i] > 0.45 else 0.0
            labels[i, 1] = 1.0 if upwind_damaged else 0.0

        X_list.append(feats)
        Y_list.append(labels)

    X_train = torch.tensor(np.array(X_list).tolist(), dtype=torch.float32)
    Y_train = torch.tensor(np.array(Y_list).tolist(), dtype=torch.float32)

    gnn = FarmGraphNet(in_features=5, hidden_dim=16)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.02)
    loss_fn = nn.BCELoss()

    for _ in range(180):
        optimizer.zero_grad()
        preds = gnn(X_train, adj)
        loss = loss_fn(preds, Y_train)
        loss.backward()
        optimizer.step()

    print(f"GNN Training Complete | Final BCE Loss: {loss.item():.4f}")

    gnn.eval()
    dummy_x = torch.randn(6, 5)
    torch.onnx.export(
        gnn,
        (dummy_x, adj),
        "models/farm_gnn.onnx",
        input_names=["node_features", "adjacency_matrix"],
        output_names=["turbine_predictions"],
        opset_version=14,
        dynamo=False,
    )

    # 2. Build & Train NLP Maintenance Log Encoder
    logs = [
        {
            "log_id": "MNT-101",
            "fault_type": "Leading-Edge Micro-Pitting",
            "notes": "High-frequency 3.8 kHz acoustic whistle synced with blade passing frequency; leading-edge gelcoat pitting repaired with polyurethane protective tape.",
            "recommended_action": "Schedule rope-access blade leading-edge resurfacing within 14 days.",
        },
        {
            "log_id": "MNT-102",
            "fault_type": "Downwind Wake Buffeting",
            "notes": "Elevated downwind turbulence and wind speed deficit caused by upwind rotor wake; blade surface intact with zero structural pitting.",
            "recommended_action": "Apply yaw-offset wake steering on upwind turbine; no blade repair needed.",
        },
        {
            "log_id": "MNT-103",
            "fault_type": "Gearbox Bearing Spalling",
            "notes": "Continuous 1.2 kHz metallic whine independent of blade passing frequency; high-speed shaft bearing replaced.",
            "recommended_action": "Inspect nacelle gearbox oil vibration spectrum.",
        },
        {
            "log_id": "MNT-104",
            "fault_type": "Nominal Operation",
            "notes": "Normal broadband aerodynamic whoosh and balanced root strain telemetry across all three blades.",
            "recommended_action": "Continue standard automated telemetry monitoring.",
        },
    ]

    logs_df = pd.DataFrame(logs)
    logs_df.to_csv("data/maintenance_logs.csv", index=False)

    vectorizer = TfidfVectorizer(max_features=64, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(
        logs_df["notes"]).toarray().astype(np.float32)

    nlp_encoder = NLPLogEncoder(vocab_size=tfidf_matrix.shape[1], embed_dim=16)
    nlp_opt = torch.optim.Adam(nlp_encoder.parameters(), lr=0.02)
    tfidf_tensor = torch.tensor(tfidf_matrix)
    target_sim = torch.eye(len(logs))

    for _ in range(150):
        nlp_opt.zero_grad()
        emb = nlp_encoder(tfidf_tensor)
        sim_matrix = torch.matmul(emb, emb.T)
        nlp_loss = nn.MSELoss()(sim_matrix, target_sim)
        nlp_loss.backward()
        nlp_opt.step()

    print(
        f"NLP Encoder Training Complete | Cosine Separation Loss: {nlp_loss.item():.4f}")

    nlp_encoder.eval()
    dummy_tfidf = torch.randn(1, tfidf_matrix.shape[1])
    torch.onnx.export(
        nlp_encoder,
        dummy_tfidf,
        "models/nlp_log_matcher.onnx",
        input_names=["tfidf_features"],
        output_names=["semantic_embedding"],
        opset_version=14,
        dynamo=False,
    )

    clean_vocab = {str(k): int(v) for k, v in vectorizer.vocabulary_.items()}
    with open("models/tfidf_vocab.json", "w", encoding="utf-8") as f:
        json.dump(clean_vocab, f, indent=2)

    print("Exported ONNX models & vocabulary cleanly.")


if __name__ == "__main__":
    train_and_export()
