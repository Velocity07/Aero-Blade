import os
import numpy as np
import pandas as pd
from scipy.io.wavfile import write as write_wav

# Ensure the output data folder exists
os.makedirs("data", exist_ok=True)


def generate_blade_audio(
    duration_sec: int,
    sample_rate: int,
    rpm: float,
    erosion_level: float,
) -> np.ndarray:
    """
    Synthesizes wind turbine microphone audio.
    - Healthy blades produce a low-frequency 'whoosh' every time a blade passes.
    - Eroded (micro-pitted) blades add a high-frequency whistle modulated by blade speed.
    """
    t = np.linspace(0, duration_sec, int(
        sample_rate * duration_sec), endpoint=False)

    # 3 blades per turbine -> Blade Passing Frequency (BPF) in Hz
    bpf = 3.0 * (rpm / 60.0)

    # Periodic pulse each time a blade sweeps past the tower
    blade_pulse = 0.5 * (1.0 + np.sin(2.0 * np.pi * bpf * t))

    # 1. Normal low-frequency aerodynamic wind whoosh
    rng = np.random.default_rng(seed=42)
    base_wind_noise = rng.normal(0, 0.15, size=t.shape)
    healthy_whoosh = base_wind_noise * (0.4 + 0.6 * blade_pulse)

    # 2. High-frequency whistle (3.8 kHz) caused by surface micro-pitting
    erosion_whistle = (
        erosion_level
        * 0.35
        * np.sin(2.0 * np.pi * 3800.0 * t)
        * (blade_pulse**2)
    )

    combined_signal = healthy_whoosh + erosion_whistle
    # Normalize safely to 16-bit PCM audio range
    normalized = np.int16(
        (combined_signal / np.max(np.abs(combined_signal))) * 32767)
    return normalized


def simulate_wind_farm() -> None:
    print("Starting Aero-Blade Physics Simulator (6 Turbines)...")

    # 6 turbines: T01-T03 in Front Row, T04-T06 in Back Row (downwind)
    turbines = [
        {"id": "T01", "row": 1, "erosion": 0.05},  # Healthy
        {"id": "T02", "row": 1, "erosion": 0.08},  # Healthy
        {"id": "T03", "row": 1, "erosion": 0.75},  # Damaged (Micro-pitting!)
        # Healthy (in T01's wind shadow)
        {"id": "T04", "row": 2, "erosion": 0.06},
        # Healthy (in T02's wind shadow)
        {"id": "T05", "row": 2, "erosion": 0.10},
        # Healthy (in T03's wind shadow)
        {"id": "T06", "row": 2, "erosion": 0.12},
    ]

    records = []
    base_wind_speed = 11.5  # meters per second (m/s)
    sample_rate = 16000     # 16 kHz audio sample rate

    for second in range(60):
        # Slight natural wind gust variation every second
        gust = np.sin(second / 8.0) * 0.6

        for t_info in turbines:
            # Row 2 loses ~18% wind speed due to the wake turbulence from Row 1
            wake_factor = 1.0 if t_info["row"] == 1 else 0.82
            local_wind = round((base_wind_speed + gust) * wake_factor, 2)

            # Rotor RPM scales with local wind speed
            rpm = round(local_wind * 1.35, 2)
            bpf_hz = round(3.0 * (rpm / 60.0), 3)

            # High-frequency acoustic energy rises sharply with blade erosion
            hf_acoustic_db = round(
                42.0 + (t_info["erosion"] * 28.0) + np.random.normal(0, 0.4), 2)

            # Traditional SCADA strain gauge barely moves yet (the blind spot!)
            scada_strain_micro = round(
                310.0 + (local_wind * 8.5) + (t_info["erosion"] * 4.0), 2)

            records.append(
                {
                    "timestamp_sec": second,
                    "turbine_id": t_info["id"],
                    "row": t_info["row"],
                    "wind_speed_ms": local_wind,
                    "rotor_rpm": rpm,
                    "blade_passing_freq_hz": bpf_hz,
                    "hf_acoustic_db": hf_acoustic_db,
                    "scada_strain_ue": scada_strain_micro,
                    "true_erosion_score": t_info["erosion"],
                }
            )

    # Save 60 seconds of farm telemetry to CSV
    df = pd.DataFrame(records)
    csv_path = "data/farm_telemetry.csv"
    df.to_csv(csv_path, index=False)

    # Generate 5-second audio clips for a healthy turbine (T01) vs damaged (T03)
    audio_healthy = generate_blade_audio(
        duration_sec=5, sample_rate=sample_rate, rpm=15.5, erosion_level=0.05)
    audio_damaged = generate_blade_audio(
        duration_sec=5, sample_rate=sample_rate, rpm=15.5, erosion_level=0.75)

    write_wav("data/T01_healthy.wav", sample_rate, audio_healthy)
    write_wav("data/T03_damaged.wav", sample_rate, audio_damaged)

    print(f"Done! Saved sensor logs to: {csv_path}")
    print("Saved audio samples to: data/T01_healthy.wav and data/T03_damaged.wav\n")
    print("--- Quick Summary (Average per Turbine) ---")
    summary = df.groupby(["turbine_id", "row"])[
        ["wind_speed_ms", "hf_acoustic_db", "scada_strain_ue"]].mean()
    print(summary.round(2))


if __name__ == "__main__":
    simulate_wind_farm()
