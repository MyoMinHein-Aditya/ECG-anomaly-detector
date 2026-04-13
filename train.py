"""
ECG Anomaly Detector - Main Training Script
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from model          import build_autoencoder, get_encoder
from data_utils     import (load_ecg5000, compute_reconstruction_errors,
                             find_threshold, find_best_threshold)
from visualisations import (
    plot_training_history, plot_reconstructions, plot_error_distributions,
    plot_confusion_matrix, plot_roc_pr_curves, plot_latent_space, plot_anomaly_timeline,
)

EPOCHS      = 100
BATCH_SIZE  = 32
LATENT_DIM  = 16
MODEL_PATH  = "outputs/ecg_autoencoder.keras"

os.makedirs("outputs", exist_ok=True)


def main():
    print("=" * 60)
    print("   ECG Anomaly Detector — Autoencoder Training (v2)")
    print("=" * 60)

    X_train, X_val, X_test_normal, X_test_anomaly, scaler = load_ecg5000(
        data_path="data/ECG5000_TRAIN.txt"
    )

    print("\nBuilding autoencoder...")
    model = build_autoencoder(sequence_length=X_train.shape[1], latent_dim=LATENT_DIM)
    model.summary()

    print(f"\nTraining for up to {EPOCHS} epochs (early stopping enabled)...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=0
        ),
    ]

    history = model.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n  Model saved -> {MODEL_PATH}")

    print("\nComputing reconstruction errors...")
    val_errors          = compute_reconstruction_errors(model, X_val)
    test_normal_errors  = compute_reconstruction_errors(model, X_test_normal)
    test_anomaly_errors = compute_reconstruction_errors(model, X_test_anomaly)

    # Use small slice of test anomalies for threshold tuning, rest for final eval
    n_tune = max(50, len(X_test_anomaly) // 4)
    tune_anomaly_errors = test_anomaly_errors[:n_tune]
    eval_anomaly_errors = test_anomaly_errors[n_tune:]
    eval_normal_errors  = test_normal_errors

    threshold = find_best_threshold(val_errors, tune_anomaly_errors)

    all_errors = np.concatenate([eval_normal_errors, eval_anomaly_errors])
    all_labels = np.concatenate([
        np.zeros(len(eval_normal_errors)),
        np.ones(len(eval_anomaly_errors))
    ])
    preds = (all_errors > threshold).astype(int)

    tp = int(np.sum((preds == 1) & (all_labels == 1)))
    fp = int(np.sum((preds == 1) & (all_labels == 0)))
    tn = int(np.sum((preds == 0) & (all_labels == 0)))
    fn = int(np.sum((preds == 0) & (all_labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / len(all_labels)

    print("\n" + "-" * 40)
    print("  Test Set Results")
    print("-" * 40)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("-" * 40)

    with open("outputs/threshold.txt", "w") as f:
        f.write(str(threshold))
    print(f"\n  Threshold saved -> outputs/threshold.txt  ({threshold:.6f})")

    print("\nGenerating visualisations...")
    encoder = get_encoder(model)

    X_combined = np.vstack([X_test_normal[:40], X_test_anomaly[:40]])
    y_combined = np.concatenate([np.zeros(40), np.ones(40)])
    idx = np.random.permutation(len(X_combined))
    X_combined, y_combined = X_combined[idx], y_combined[idx]

    plot_training_history(history)
    plot_reconstructions(model, X_test_normal, X_test_anomaly, n=3)
    plot_error_distributions(test_normal_errors, test_anomaly_errors, threshold)
    plot_confusion_matrix(eval_normal_errors, eval_anomaly_errors, threshold)
    plot_roc_pr_curves(eval_normal_errors, eval_anomaly_errors)
    plot_latent_space(encoder, X_test_normal, X_test_anomaly)
    plot_anomaly_timeline(model, X_combined, y_combined, threshold, n_signals=80)

    print("\n  All visualisations saved to ./outputs/")
    print("  Training complete!\n")


if __name__ == "__main__":
    main()
