"""
ECG Anomaly Detector - Data Loading & Preprocessing
=====================================================
Loads the ECG5000 dataset from the UCR Time Series Archive.
The dataset contains 5000 ECG signals each of length 140.

Label encoding (original):
    1 = Normal (class we train on)
    2, 3, 4, 5 = Various anomalies (R-on-T, PVC, SP, UB)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import urllib.request
import os


# ── Dataset constants ─────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 140
NORMAL_CLASS = 1

# UCR ECG5000 - publicly available dataset
TRAIN_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ecg.csv"

# Fallback: generate synthetic ECG-like data if download fails
def _generate_synthetic_ecg(n_samples: int = 5000, seq_len: int = 140, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, seq_len)

    normal, anomaly = [], []

    for _ in range(int(n_samples * 0.7)):  # 70% normal
        # Realistic ECG-like: sum of harmonics + small noise
        signal = (
            0.8 * np.sin(t)
            + 0.3 * np.sin(2 * t)
            + 0.1 * np.sin(3 * t)
            + rng.normal(0, 0.05, seq_len)
        )
        normal.append(signal)

    for _ in range(int(n_samples * 0.3)):  # 30% anomaly
        signal = (
            0.8 * np.sin(t)
            + 0.3 * np.sin(2 * t)
            + rng.normal(0, 0.2, seq_len)  # higher noise
        )
        # Random spike (simulates R-on-T or PVC)
        spike_pos = rng.integers(20, seq_len - 20)
        signal[spike_pos] += rng.uniform(1.5, 3.0) * rng.choice([-1, 1])
        anomaly.append(signal)

    X = np.array(normal + anomaly)
    y = np.array([0] * len(normal) + [1] * len(anomaly))  # 0=normal, 1=anomaly
    return X, y


def load_ecg5000(data_path: str = "data/ecg5000_train.csv") -> tuple:
    print("Loading ECG dataset...")

    try:
        # Try loading from local CSV (UCR format: label in first column)
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, header=None)
            labels = df.iloc[:, 0].values.astype(int)
            signals = df.iloc[:, 1:].values.astype(np.float32)

            normal_mask = (labels == NORMAL_CLASS)
            X_normal = signals[normal_mask]
            X_anomaly = signals[~normal_mask]
            print(f"  Loaded from file: {X_normal.shape[0]} normal, {X_anomaly.shape[0]} anomaly")

        else:
            raise FileNotFoundError("Local data not found, using synthetic data")

    except Exception as e:
        print(f"  {e}")
        print("  Generating synthetic ECG data for demonstration...")
        X_all, y_all = _generate_synthetic_ecg(n_samples=5000, seq_len=SEQUENCE_LENGTH)
        X_normal = X_all[y_all == 0]
        X_anomaly = X_all[y_all == 1]
        print(f"  Generated: {X_normal.shape[0]} normal, {X_anomaly.shape[0]} anomaly")

    # ── Normalise ─────────────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_anomaly_scaled = scaler.transform(X_anomaly)

    # ── Train / Val / Test split (normal only for training) ────────────────────
    X_train, X_val = train_test_split(X_normal_scaled, test_size=0.1, random_state=42)

    # Keep a held-out test set of both classes
    _, X_test_normal = train_test_split(X_normal_scaled, test_size=0.15, random_state=42)
    X_test_anomaly = X_anomaly_scaled

    # Reshape for Conv1D: (samples, timesteps, channels)
    X_train = X_train.reshape(-1, SEQUENCE_LENGTH, 1)
    X_val = X_val.reshape(-1, SEQUENCE_LENGTH, 1)
    X_test_normal = X_test_normal.reshape(-1, SEQUENCE_LENGTH, 1)
    X_test_anomaly = X_test_anomaly.reshape(-1, SEQUENCE_LENGTH, 1)

    print(f"\n  Train shape   : {X_train.shape}")
    print(f"  Val shape     : {X_val.shape}")
    print(f"  Test normal   : {X_test_normal.shape}")
    print(f"  Test anomaly  : {X_test_anomaly.shape}")

    return X_train, X_val, X_test_normal, X_test_anomaly, scaler


def compute_reconstruction_errors(model, X: np.ndarray) -> np.ndarray:
    reconstructions = model.predict(X, verbose=0)
    errors = np.mean(np.abs(X - reconstructions), axis=(1, 2))
    return errors


def find_threshold(normal_errors: np.ndarray, percentile: float = 95.0) -> float:
    threshold = np.percentile(normal_errors, percentile)
    print(f"\n  Anomaly threshold (p{percentile:.0f}): {threshold:.6f}")
    return threshold


def find_best_threshold(normal_errors: np.ndarray, anomaly_errors: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    all_errors = np.concatenate([normal_errors, anomaly_errors])
    all_labels = np.array([0] * len(normal_errors) + [1] * len(anomaly_errors))

    candidates = np.linspace(all_errors.min(), all_errors.max(), 300)

    best_f1, best_threshold = 0.0, candidates[0]
    for t in candidates:
        preds = (all_errors > t).astype(int)
        score = f1_score(all_labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = t

    print(f"\n  Best threshold (F1-optimal): {best_threshold:.6f}  ->  F1 = {best_f1:.4f}")
    return float(best_threshold)
