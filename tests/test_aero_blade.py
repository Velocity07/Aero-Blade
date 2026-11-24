import importlib.util
import sys
from pathlib import Path
from fastapi.testclient import TestClient

MAIN_PATH = Path(__file__).resolve(
).parents[1] / "services" / "graphql-gateway" / "main.py"
spec = importlib.util.spec_from_file_location("aero_gateway_main", MAIN_PATH)
gateway_main = importlib.util.module_from_spec(spec)
sys.modules["aero_gateway_main"] = gateway_main
spec.loader.exec_module(gateway_main)

client = TestClient(gateway_main.app)


def test_dsp_acoustic_kurtosis_and_spl():
    hf_healthy, kurt_healthy, _, _ = gateway_main.synthesize_and_analyze_acoustics(
        rpm=15.5, erosion=0.05, seed=1
    )
    hf_pitted, kurt_pitted, bins, wav_bytes = gateway_main.synthesize_and_analyze_acoustics(
        rpm=15.5, erosion=0.85, seed=1
    )

    assert hf_pitted > hf_healthy * 2.5
    assert kurt_pitted > kurt_healthy + 2.0
    assert len(bins) == 64
    assert wav_bytes[:4] == b"RIFF"


def test_graphql_and_rest_endpoints():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "online"

    gql = client.post(
        "/graphql",
        json={
            "query": "{ farmSummary { criticalCount turbines { turbineId pittingProb spectralKurtosis } } }"
        },
    )
    assert gql.status_code == 200
    data = gql.json()["data"]["farmSummary"]
    assert data["criticalCount"] == 1
    assert len(data["turbines"]) == 6

    wav = client.get("/api/audio/T03.wav")
    assert wav.status_code == 200
    assert wav.headers["content-type"] == "audio/wav"
