import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis


def analyze_audio_file(wav_path: str) -> dict:
    """
    Applies Digital Signal Processing (DSP) to a turbine microphone recording
    to isolate micro-pitting acoustic signatures and verify rotor synchronization.
    """
    sample_rate, raw_audio = wavfile.read(wav_path)

    # Convert 16-bit integer audio into normalized floating-point values (-1.0 to 1.0)
    signal = raw_audio.astype(np.float64) / 32768.0

    # 1. Bandpass Filter (3,200 Hz - 4,400 Hz) to isolate blade micro-pitting whistle
    nyquist = 0.5 * sample_rate
    low_cut = 3200.0 / nyquist
    high_cut = 4400.0 / nyquist
    b, a = butter(N=4, Wn=[low_cut, high_cut], btype="band")
    filtered_signal = filtfilt(b, a, signal)

    # 2. High-Frequency Energy Ratio (Filtered whistle power vs. total audio power)
    total_power = np.mean(signal**2) + 1e-12
    hf_power = np.mean(filtered_signal**2)
    hf_energy_ratio = hf_power / total_power

    # 3. Signal Kurtosis (Measures sharp periodic bursts vs. steady background wind)
    burst_kurtosis = float(kurtosis(filtered_signal))

    # 4. Hilbert Envelope + FFT to recover the Blade Passing Frequency (BPF)
    analytic_signal = hilbert(filtered_signal)
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope -= np.mean(amplitude_envelope)  # Remove DC offset

    n_samples = len(amplitude_envelope)
    fft_freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    fft_magnitudes = np.abs(np.fft.rfft(amplitude_envelope)) / n_samples

    # Search inside realistic 3-blade rotor BPF range (0.4 Hz to 2.0 Hz)
    valid_bpf_mask = (fft_freqs >= 0.4) & (fft_freqs <= 2.0)
    detected_bpf_hz = float(
        fft_freqs[valid_bpf_mask][np.argmax(fft_magnitudes[valid_bpf_mask])])
    estimated_rpm = (detected_bpf_hz / 3.0) * 60.0

    # Compute normalized Acoustic Erosion Risk Score (0.0 = Healthy, 1.0 = Severe Pitting)
    risk_score = np.clip((hf_energy_ratio - 0.15) * 1.6 +
                         (burst_kurtosis * 0.08), 0.0, 1.0)
    status = "CRITICAL - MICRO-PITTING" if risk_score > 0.50 else "HEALTHY"

    return {
        "file": wav_path,
        "hf_energy_ratio": round(float(hf_energy_ratio), 3),
        "burst_kurtosis": round(burst_kurtosis, 3),
        "detected_bpf_hz": round(detected_bpf_hz, 2),
        "estimated_rpm": round(estimated_rpm, 1),
        "erosion_risk_score": round(float(risk_score), 3),
        "diagnosis": status,
    }


def extract_farm_node_features() -> None:
    """
    Runs acoustic DSP diagnostics on audio clips and prepares clean per-turbine
    feature vectors for Milestone 3 (the Farm-Wide Graph Neural Network).
    """
    print("=== Aero-Blade Edge Acoustic DSP Engine ===\n")

    audio_files = ["data/T01_healthy.wav", "data/T03_damaged.wav"]
    results = [analyze_audio_file(path) for path in audio_files]
    report_df = pd.DataFrame(results)

    print("--- Acoustic Waveform Diagnostic Report ---")
    print(report_df.to_string(index=False))
    print("\n")

    # Aggregate 60-second telemetry into node-level features for all 6 turbines
    telemetry = pd.read_csv("data/farm_telemetry.csv")
    node_features = (
        telemetry.groupby(["turbine_id", "row"])
        .agg(
            avg_wind_ms=("wind_speed_ms", "mean"),
            avg_rpm=("rotor_rpm", "mean"),
            avg_bpf_hz=("blade_passing_freq_hz", "mean"),
            avg_hf_db=("hf_acoustic_db", "mean"),
            avg_strain_ue=("scada_strain_ue", "mean"),
            true_erosion=("true_erosion_score", "first"),
        )
        .reset_index()
    )

    # Calculate acoustic-to-strain anomaly index (high acoustic energy relative to structural strain)
    node_features["acoustic_anomaly_index"] = (
        (node_features["avg_hf_db"] - 42.0) /
        (node_features["avg_strain_ue"] / 400.0)
    ).round(2)

    output_path = "data/dsp_node_features.csv"
    node_features.to_csv(output_path, index=False)

    print(f"Saved 6-turbine node feature matrix to: {output_path}")
    print(node_features.round(2).to_string(index=False))


if __name__ == "__main__":
    extract_farm_node_features()
