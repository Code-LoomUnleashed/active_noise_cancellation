"""
Active Noise Cancellation (ANC) Demo
-----------------------------------
A software simulation of feedforward active noise cancellation using an
adaptive LMS (Least Mean Squares) filter.

This script:
1. Creates a reference noise signal and a primary noise signal.
2. Adapts an FIR filter to estimate the noise reaching the error microphone.
3. Subtracts the estimate to produce a noise-reduced output.
4. Reports RMS levels and saves a comparison plot.

Requirements:
    pip install numpy matplotlib

Run:
    python active_noise_cancellation.py

Note:
This is an offline simulation, not a guaranteed real-time hardware ANC system.
Real ANC needs synchronized microphones/speakers, low-latency audio I/O, and
careful acoustic-path calibration. This basic LMS demo assumes the reference
signal is correlated with the noise at the error microphone.
"""

import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window=100):
    """Smooth a signal for visualization."""
    window = max(1, int(window))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def run_anc(
    sample_rate=16000,
    duration_seconds=3.0,
    filter_length=64,
    step_size=0.01,
    random_seed=7,
):
    """Simulate ANC with an adaptive LMS FIR filter."""
    rng = np.random.default_rng(random_seed)
    n_samples = int(sample_rate * duration_seconds)
    t = np.arange(n_samples) / sample_rate

    # Reference microphone signal: a mixture of tonal and broadband noise.
    reference = (
        0.55 * np.sin(2 * np.pi * 120 * t)
        + 0.30 * np.sin(2 * np.pi * 240 * t + 0.4)
        + 0.18 * rng.normal(size=n_samples)
    )

    # Simulated acoustic path from the noise source to the error microphone.
    # A short delayed/filtered version of the reference plus small unrelated noise.
    path = np.array([0.0, 0.65, 0.28, -0.12, 0.06])
    primary_noise = np.convolve(reference, path, mode="full")[:n_samples]
    primary_noise += 0.025 * rng.normal(size=n_samples)

    # Adaptive FIR filter (LMS).
    weights = np.zeros(filter_length, dtype=float)
    delay_line = np.zeros(filter_length, dtype=float)
    estimated_noise = np.zeros(n_samples, dtype=float)
    error = np.zeros(n_samples, dtype=float)

    # Normalized LMS is more robust to changing input amplitude.
    epsilon = 1e-8
    for n in range(n_samples):
        delay_line[1:] = delay_line[:-1]
        delay_line[0] = reference[n]

        estimated_noise[n] = np.dot(weights, delay_line)
        error[n] = primary_noise[n] - estimated_noise[n]

        norm = np.dot(delay_line, delay_line) + epsilon
        weights += (step_size / norm) * error[n] * delay_line

    return t, primary_noise, estimated_noise, error, sample_rate


def rms(signal):
    """Return root-mean-square signal amplitude."""
    return float(np.sqrt(np.mean(np.square(signal))))


def main():
    t, primary, anti_noise_estimate, residual, sample_rate = run_anc()

    # Avoid judging adaptation only by the initial transient.
    start = int(0.5 * sample_rate)
    before_rms = rms(primary[start:])
    after_rms = rms(residual[start:])
    reduction_db = 20 * np.log10((before_rms + 1e-12) / (after_rms + 1e-12))

    print("Active Noise Cancellation — LMS simulation")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Input noise RMS:  {before_rms:.4f}")
    print(f"Residual RMS:     {after_rms:.4f}")
    print(f"RMS reduction:    {reduction_db:.2f} dB")

    # Plot a short section so the waveforms are readable.
    shown_samples = min(len(t), int(0.08 * sample_rate))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(t[:shown_samples], primary[:shown_samples], linewidth=0.8)
    axes[0].set_title("Primary noise at error microphone")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        t[:shown_samples],
        anti_noise_estimate[:shown_samples],
        linewidth=0.8,
    )
    axes[1].set_title("Adaptive filter's estimated noise / anti-noise reference")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t[:shown_samples], residual[:shown_samples], linewidth=0.8)
    axes[2].set_title("Residual signal after cancellation")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    output_plot = "anc_results.png"
    fig.savefig(output_plot, dpi=150)
    print(f"Plot saved to: {output_plot}")
    plt.show()


if __name__ == "__main__":
    main()
