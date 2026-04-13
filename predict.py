"""
ECG Anomaly Detector - Inference
==================================
Load a trained model and classify new ECG signals.

Usage:
    python predict.py --input your_ecg_file.csv --threshold 0.05

The CSV should have one ECG signal per row (140 values per row, no header).
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os


THRESHOLD_DEFAULT = 0.05   # override with --threshold if you retrained the model


def load_model(model_path: str = "outputs/ecg_autoencoder.keras"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run train.py first to train and save the model."
        )
    print(f"Loading model from {model_path}...")
    return tf.keras.models.load_model(model_path)


def predict(model, signals: np.ndarray, threshold: float) -> dict:
    reconstructions = model.predict(signals, verbose=0)
    errors = np.mean(np.abs(signals - reconstructions), axis=(1, 2))
    predictions = (errors > threshold).astype(int)
    labels = ["ANOMALY" if p == 1 else "NORMAL" for p in predictions]

    return {
        "errors":      errors,
        "predictions": predictions,
        "labels":      labels,
    }


def plot_prediction(model, signal: np.ndarray, error: float,
                    label: str, threshold: float, idx: int = 0):
    """Visualise a single prediction with its reconstruction."""
    recon = model.predict(signal[np.newaxis], verbose=0)[0, :, 0]
    orig  = signal[:, 0]
    t     = np.arange(len(orig))

    color = "#E8624C" if label == "ANOMALY" else "#4C9BE8"

    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0F1117")
    ax.set_facecolor("#1A1D27")

    ax.plot(t, orig,  color=color,    lw=2,   label="Input Signal")
    ax.plot(t, recon, color="#F5A623", lw=1.5, linestyle="--", label="Reconstruction")
    ax.fill_between(t, orig, recon, alpha=0.2, color="#F5A623")

    ax.set_title(f"Signal #{idx}  →  {label}  (error={error:.5f}, threshold={threshold:.5f})",
                 color="#E8EAF0", fontsize=13)
    ax.set_xlabel("Timestep", color="#E8EAF0")
    ax.set_ylabel("Amplitude", color="#E8EAF0")
    ax.tick_params(colors="#E8EAF0")
    ax.spines[["top","right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(facecolor="#1A1D27", labelcolor="#E8EAF0")

    out = f"outputs/prediction_{idx:03d}_{label.lower()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out}")


def main():
    parser = argparse.ArgumentParser(description="Predict ECG anomalies")
    parser.add_argument("--input",     type=str,   default=None,
                        help="Path to CSV file (one signal per row, 140 columns)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                        help="Reconstruction error threshold for anomaly classification")
    parser.add_argument("--model",     type=str,   default="outputs/ecg_autoencoder.keras",
                        help="Path to saved .keras model file")
    parser.add_argument("--plot",      action="store_true",
                        help="Generate prediction plots for each signal")
    args = parser.parse_args()

    model = load_model(args.model)
    os.makedirs("outputs", exist_ok=True)

    # ── Load signals ──────────────────────────────────────────────────────────
    if args.input:
        df = pd.read_csv(args.input, header=None)
        signals = df.values.astype(np.float32)
    else:
        print("No --input provided. Generating 10 synthetic test signals...")
        from data_utils import _generate_synthetic_ecg
        from sklearn.preprocessing import MinMaxScaler
        X, _ = _generate_synthetic_ecg(n_samples=10, seq_len=140)
        scaler = MinMaxScaler()
        signals = scaler.fit_transform(X)

    signals = signals.reshape(-1, 140, 1)

    # ── Predict ───────────────────────────────────────────────────────────────
    results = predict(model, signals, args.threshold)

    print(f"\n{'─'*50}")
    print(f"  Results  (threshold = {args.threshold:.5f})")
    print(f"{'─'*50}")
    for i, (err, lbl) in enumerate(zip(results["errors"], results["labels"])):
        marker = "⚠" if lbl == "ANOMALY" else "✓"
        print(f"  Signal {i:3d}:  error={err:.5f}  {marker}  {lbl}")

    n_anomalies = sum(results["predictions"])
    print(f"\n  Total: {len(signals)} signals — {n_anomalies} anomalies detected "
          f"({100*n_anomalies/len(signals):.1f}%)")

    # ── Optional plots ────────────────────────────────────────────────────────
    if args.plot:
        print("\nGenerating prediction plots...")
        for i, (sig, err, lbl) in enumerate(zip(signals, results["errors"], results["labels"])):
            plot_prediction(model, sig, err, lbl, args.threshold, idx=i)

    print("\nDone.")


if __name__ == "__main__":
    main()
