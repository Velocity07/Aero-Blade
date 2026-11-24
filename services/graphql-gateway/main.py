import io
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
import strawberry
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from scipy.io.wavfile import write as write_wav
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis
from sklearn.feature_extraction.text import TfidfVectorizer
from strawberry.fastapi import GraphQLRouter

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

gnn_session = ort.InferenceSession(str(MODELS_DIR / "farm_gnn.onnx"))
nlp_session = ort.InferenceSession(str(MODELS_DIR / "nlp_log_matcher.onnx"))


def build_normalized_adjacency() -> np.ndarray:
    adj = np.eye(6, dtype=np.float32)
    for upwind, downwind in [(0, 3), (1, 4), (2, 5)]:
        adj[downwind, upwind] = 0.65
    for src, dst in [(0, 1), (1, 0), (1, 2), (2, 1), (3, 4), (4, 3), (4, 5), (5, 4)]:
        adj[dst, src] = 0.20
    return adj / adj.sum(axis=1, keepdims=True)


def synthesize_and_analyze_acoustics(
    rpm: float, erosion: float, seed: int
) -> tuple[float, float, list[dict], bytes]:
    """
    Synthesizes 2.0s of 16 kHz turbine microphone audio with impulsive blade-pass
    micro-pitting bursts, applies a 4th-order Butterworth bandpass filter (3.2-4.4 kHz),
    computes a 64-bin rfft Power Spectral Density, and encodes playable 16-bit WAV bytes.
    """
    sample_rate = 16000
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(
        sample_rate * duration_sec), endpoint=False)
    bpf = 3.0 * (rpm / 60.0)

    # Smooth rotor sweep for ambient wind + sharp impulsive envelope for leading-edge pitting
    blade_sweep = 0.5 * (1.0 + np.sin(2.0 * np.pi * bpf * t))
    impulsive_burst = np.maximum(0.0, np.sin(2.0 * np.pi * bpf * t)) ** 12

    rng = np.random.default_rng(seed=seed)
    wind_noise = rng.normal(0.0, 0.14, size=t.shape) * \
        (0.45 + 0.55 * blade_sweep)
    pitting_tone = (
        erosion * 0.52 * np.sin(2.0 * np.pi * 3800.0 * t) * impulsive_burst
    )
    signal = wind_noise + pitting_tone

    # 4th-order Butterworth Bandpass Filter (3,200 Hz - 4,400 Hz)
    nyq = 0.5 * sample_rate
    b, a = butter(N=4, Wn=[3200.0 / nyq, 4400.0 / nyq], btype="band")
    filtered = filtfilt(b, a, signal)

    hf_ratio = float(np.mean(filtered**2) / (np.mean(signal**2) + 1e-12))
    spec_kurtosis = float(kurtosis(filtered))

    # Compute 64-bin Power Spectral Density (0.0 to 8.0 kHz) using Hann window + rfft
    windowed = signal * np.hanning(len(signal))
    fft_vals = np.abs(np.fft.rfft(windowed)) / len(signal)
    psd_db = 20.0 * np.log10(fft_vals + 1e-7) + 95.0

    bins_per_group = len(psd_db) // 64
    spectrum_bins = []
    for i in range(64):
        chunk = psd_db[i * bins_per_group: (i + 1) * bins_per_group]
        freq_khz = round((i + 0.5) * (8.0 / 64.0), 3)
        spectrum_bins.append(
            {
                "freq_khz": freq_khz,
                "power_db": round(float(np.clip(np.max(chunk), 12.0, 88.0)), 2),
            }
        )

    # Encode 16-bit PCM WAV buffer for live browser playback
    peak = np.max(np.abs(signal)) + 1e-9
    pcm16 = np.int16((signal / peak) * 30000)
    wav_buf = io.BytesIO()
    write_wav(wav_buf, sample_rate, pcm16)

    return round(hf_ratio, 3), round(spec_kurtosis, 3), spectrum_bins, wav_buf.getvalue()


def build_30d_trajectory(
    local_wind: float, current_erosion: float, seed: int
) -> list[dict]:
    rng = np.random.default_rng(seed=seed + 100)
    history = []
    initial_erosion = min(0.06, current_erosion)
    for day_idx in range(30):
        progress = (day_idx / 29.0) ** 1.65
        e_day = initial_erosion + \
            (current_erosion - initial_erosion) * progress
        wind_day = local_wind + float(rng.normal(0.0, 0.25))
        hf_db = 42.0 + (e_day * 28.0) + float(rng.normal(0.0, 0.35))
        strain_ue = 310.0 + (wind_day * 8.5) + (e_day * 4.0)
        label = "NOW" if day_idx == 29 else f"D-{29 - day_idx:02d}"
        history.append(
            {
                "day_label": label,
                "acoustic_db": round(hf_db, 2),
                "strain_ue": round(strain_ue, 1),
            }
        )
    return history


def run_live_onnx_pipeline(
    base_wind_ms: float = 11.5,
    damaged_turbine_id: str = "T03",
    erosion_level: float = 0.75,
) -> dict:
    turbines_cfg = [
        {"id": "T01", "row": 1, "base_erosion": 0.05},
        {"id": "T02", "row": 1, "base_erosion": 0.08},
        {"id": "T03", "row": 1, "base_erosion": 0.06},
        {"id": "T04", "row": 2, "base_erosion": 0.06},
        {"id": "T05", "row": 2, "base_erosion": 0.10},
        {"id": "T06", "row": 2, "base_erosion": 0.11},
    ]

    target_clean = damaged_turbine_id.upper()
    rng = np.random.default_rng(seed=42)
    node_rows = []
    dsp_extras = {}

    for idx, t in enumerate(turbines_cfg):
        erosion = erosion_level if t["id"] == target_clean else t["base_erosion"]
        wake_factor = 1.0 if t["row"] == 1 else 0.82
        local_wind = round(base_wind_ms * wake_factor, 2)
        rpm = round(local_wind * 1.35, 2)
        bpf_hz = round(3.0 * (rpm / 60.0), 3)
        hf_db = round(42.0 + (erosion * 28.0) + float(rng.normal(0, 0.2)), 2)
        strain_ue = round(310.0 + (local_wind * 8.5) + (erosion * 4.0), 2)
        anomaly_idx = round((hf_db - 42.0) / (strain_ue / 400.0), 2)

        hf_ratio, spec_kurt, spectrum_bins, wav_bytes = synthesize_and_analyze_acoustics(
            rpm=rpm, erosion=erosion, seed=10 + idx
        )
        history_30d = build_30d_trajectory(
            local_wind=local_wind, current_erosion=erosion, seed=20 + idx
        )
        dsp_extras[t["id"]] = {
            "hf_energy_ratio": hf_ratio,
            "spectral_kurtosis": spec_kurt,
            "spectrum_bins": spectrum_bins,
            "history_30d": history_30d,
            "wav_bytes": wav_bytes,
        }

        node_rows.append(
            {
                "turbine_id": t["id"],
                "row": t["row"],
                "avg_wind_ms": local_wind,
                "avg_rpm": rpm,
                "avg_bpf_hz": bpf_hz,
                "avg_hf_db": hf_db,
                "avg_strain_ue": strain_ue,
                "true_erosion": erosion,
                "acoustic_anomaly_index": anomaly_idx,
            }
        )

    nodes_df = pd.DataFrame(node_rows)
    nodes_df.to_csv(DATA_DIR / "dsp_node_features.csv", index=False)

    features = np.column_stack(
        [
            nodes_df["avg_wind_ms"].values / 15.0,
            nodes_df["avg_rpm"].values / 20.0,
            nodes_df["avg_bpf_hz"].values / 1.5,
            nodes_df["avg_hf_db"].values / 80.0,
            nodes_df["acoustic_anomaly_index"].values / 25.0,
        ]
    ).astype(np.float32)

    logs_df = pd.read_csv(DATA_DIR / "maintenance_logs.csv")
    with open(MODELS_DIR / "tfidf_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vectorizer = TfidfVectorizer(vocabulary=vocab, stop_words="english")
    log_tfidf = vectorizer.fit_transform(
        logs_df["notes"]).toarray().astype(np.float32)

    t_onnx_start = time.perf_counter()
    gnn_outputs = gnn_session.run(
        None,
        {"node_features": features, "adjacency_matrix": build_normalized_adjacency()},
    )[0]
    log_embeddings = np.vstack(
        [
            nlp_session.run(None, {"tfidf_features": log_tfidf[i: i + 1]})[0]
            for i in range(len(log_tfidf))
        ]
    )

    diagnostics = []
    for idx, row in nodes_df.iterrows():
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
        similarities = (log_embeddings @ query_emb.T).squeeze()
        best_idx = int(np.argmax(similarities))
        matched_log = logs_df.iloc[best_idx]

        diagnostics.append(
            {
                "turbine_id": row["turbine_id"],
                "row": int(row["row"]),
                "pitting_prob": round(pitting_prob, 3),
                "wake_impact_prob": round(wake_prob, 3),
                "status": status,
                "matched_log_id": matched_log["log_id"],
                "nlp_similarity": round(float(similarities[best_idx]), 3),
                "recommended_action": matched_log["recommended_action"],
            }
        )

    onnx_ms = round((time.perf_counter() - t_onnx_start) * 1000.0, 2)
    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(DATA_DIR / "farm_diagnostics.csv", index=False)

    return {
        "nodes_df": nodes_df,
        "diag_df": diag_df,
        "dsp_extras": dsp_extras,
        "onnx_inference_ms": onnx_ms,
        "scenario": {
            "base_wind_ms": round(float(base_wind_ms), 1),
            "damaged_turbine_id": target_clean,
            "erosion_level": round(float(erosion_level), 2),
        },
    }


LATEST_STATE = run_live_onnx_pipeline()


@strawberry.type
class ScenarioConfig:
    base_wind_ms: float
    damaged_turbine_id: str
    erosion_level: float


@strawberry.type
class SpectrumPoint:
    freq_khz: float
    power_db: float


@strawberry.type
class HistoryPoint:
    day_label: str
    acoustic_db: float
    strain_ue: float


@strawberry.type
class TurbineNode:
    turbine_id: str
    row: int
    avg_wind_ms: float
    avg_rpm: float
    avg_bpf_hz: float
    avg_hf_db: float
    avg_strain_ue: float
    acoustic_anomaly_index: float
    hf_energy_ratio: float
    spectral_kurtosis: float
    pitting_prob: float
    wake_impact_prob: float
    status: str
    matched_log_id: str
    nlp_similarity: float
    recommended_action: str
    upwind_turbine_id: Optional[str]
    downwind_turbine_id: Optional[str]
    spectrum_bins: list[SpectrumPoint]
    history_30d: list[HistoryPoint]


@strawberry.type
class WindFarmSummary:
    farm_name: str
    total_turbines: int
    critical_count: int
    onnx_inference_ms: float
    active_scenario: ScenarioConfig
    turbines: list[TurbineNode]


def serialize_farm_summary(row_filter: Optional[int] = None) -> WindFarmSummary:
    merged = pd.merge(
        LATEST_STATE["nodes_df"],
        LATEST_STATE["diag_df"],
        on=["turbine_id", "row"],
    )
    dsp_extras = LATEST_STATE["dsp_extras"]
    scen = LATEST_STATE["scenario"]

    upwind_map = {"T04": "T01", "T05": "T02", "T06": "T03"}
    downwind_map = {"T01": "T04", "T02": "T05", "T03": "T06"}

    all_turbines: list[TurbineNode] = []
    for _, r in merged.iterrows():
        tid = str(r["turbine_id"])
        extra = dsp_extras[tid]
        all_turbines.append(
            TurbineNode(
                turbine_id=tid,
                row=int(r["row"]),
                avg_wind_ms=float(r["avg_wind_ms"]),
                avg_rpm=float(r["avg_rpm"]),
                avg_bpf_hz=float(r["avg_bpf_hz"]),
                avg_hf_db=float(r["avg_hf_db"]),
                avg_strain_ue=float(r["avg_strain_ue"]),
                acoustic_anomaly_index=float(r["acoustic_anomaly_index"]),
                hf_energy_ratio=float(extra["hf_energy_ratio"]),
                spectral_kurtosis=float(extra["spectral_kurtosis"]),
                pitting_prob=float(r["pitting_prob"]),
                wake_impact_prob=float(r["wake_impact_prob"]),
                status=str(r["status"]),
                matched_log_id=str(r["matched_log_id"]),
                nlp_similarity=float(r["nlp_similarity"]),
                recommended_action=str(r["recommended_action"]),
                upwind_turbine_id=upwind_map.get(tid),
                downwind_turbine_id=downwind_map.get(tid),
                spectrum_bins=[
                    SpectrumPoint(freq_khz=p["freq_khz"],
                                  power_db=p["power_db"])
                    for p in extra["spectrum_bins"]
                ],
                history_30d=[
                    HistoryPoint(
                        day_label=h["day_label"],
                        acoustic_db=h["acoustic_db"],
                        strain_ue=h["strain_ue"],
                    )
                    for h in extra["history_30d"]
                ],
            )
        )

    critical_count = sum(1 for t in all_turbines if "CRITICAL" in t.status)
    filtered = (
        [t for t in all_turbines if t.row == row_filter]
        if row_filter is not None
        else all_turbines
    )

    return WindFarmSummary(
        farm_name="North-Sea-Alpha-01",
        total_turbines=len(all_turbines),
        critical_count=critical_count,
        onnx_inference_ms=float(LATEST_STATE["onnx_inference_ms"]),
        active_scenario=ScenarioConfig(
            base_wind_ms=scen["base_wind_ms"],
            damaged_turbine_id=scen["damaged_turbine_id"],
            erosion_level=scen["erosion_level"],
        ),
        turbines=filtered,
    )


@strawberry.type
class Query:
    @strawberry.field
    def farm_summary(self, row: Optional[int] = None) -> WindFarmSummary:
        return serialize_farm_summary(row_filter=row)

    @strawberry.field
    def turbine(self, turbine_id: str) -> Optional[TurbineNode]:
        summary = serialize_farm_summary()
        for t in summary.turbines:
            if t.turbine_id.upper() == turbine_id.upper():
                return t
        return None


@strawberry.type
class Mutation:
    @strawberry.mutation
    def simulate_live_scenario(
        self,
        base_wind_ms: float = 11.5,
        damaged_turbine_id: str = "T03",
        erosion_level: float = 0.75,
    ) -> WindFarmSummary:
        global LATEST_STATE
        LATEST_STATE = run_live_onnx_pipeline(
            base_wind_ms=base_wind_ms,
            damaged_turbine_id=damaged_turbine_id,
            erosion_level=erosion_level,
        )
        return serialize_farm_summary()


schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)

app = FastAPI(
    title="Aero-Blade GraphQL Topology Gateway",
    description="Predictive Aero-Acoustics & Spatial GNN Inference Microservice",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graphql_app, prefix="/graphql")


@app.get("/")
def serve_dashboard() -> FileResponse:
    return FileResponse(ROOT_DIR / "frontend" / "index.html")


@app.get("/api/audio/{turbine_id}.wav")
def stream_turbine_audio(turbine_id: str) -> Response:
    tid = turbine_id.upper()
    extra = LATEST_STATE["dsp_extras"].get(tid)
    if not extra:
        return Response(status_code=404, content=f"Unknown turbine {tid}")
    return Response(content=extra["wav_bytes"], media_type="audio/wav")


@app.get("/api/telemetry.csv")
def export_telemetry_csv() -> Response:
    merged = pd.merge(
        LATEST_STATE["nodes_df"],
        LATEST_STATE["diag_df"],
        on=["turbine_id", "row"],
    )
    return Response(
        content=merged.to_csv(index=False),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=aero_blade_telemetry.csv"},
    )


@app.get("/health")
def health_check() -> dict:
    return {
        "service": "aero-blade-graphql-gateway",
        "status": "online",
        "onnx_inference_ms": LATEST_STATE["onnx_inference_ms"],
        "models": ["farm_gnn.onnx", "nlp_log_matcher.onnx"],
    }
