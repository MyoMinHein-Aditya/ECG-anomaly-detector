"""
ECG Anomaly Detector - Visualisations
=======================================
All plots are built with Seaborn + Matplotlib.
Each function saves a PNG and also returns the Figure object.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
import os

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
PALETTE = {
    "normal":  "#4C9BE8",   # blue
    "anomaly": "#E8624C",   # coral-red
    "accent":  "#F5A623",   # amber
    "bg":      "#0F1117",
    "surface": "#1A1D27",
    "text":    "#E8EAF0",
}
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved → {path}")
    return path


# ── 1. Training history ───────────────────────────────────────────────────────
def plot_training_history(history) -> plt.Figure:
    """Plot training & validation loss curves over epochs."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    epochs = range(1, len(history.history["loss"]) + 1)
    ax.plot(epochs, history.history["loss"],     color=PALETTE["normal"],  lw=2, label="Train Loss")
    ax.plot(epochs, history.history["val_loss"], color=PALETTE["accent"],  lw=2, linestyle="--", label="Val Loss")

    ax.set_title("Autoencoder Training Loss (MAE)", color=PALETTE["text"], fontsize=14, pad=12)
    ax.set_xlabel("Epoch", color=PALETTE["text"])
    ax.set_ylabel("MAE Loss", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])

    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    _save(fig, "01_training_history.png")
    return fig


# ── 2. Signal reconstruction comparison ──────────────────────────────────────
def plot_reconstructions(model, X_normal: np.ndarray, X_anomaly: np.ndarray, n: int = 3) -> plt.Figure:
    """
    Show original vs reconstructed signals for normal and anomalous ECGs side by side.
    Normal signals should reconstruct well; anomalous ones will show high residuals.
    """
    recon_normal  = model.predict(X_normal[:n],  verbose=0)
    recon_anomaly = model.predict(X_anomaly[:n], verbose=0)

    fig, axes = plt.subplots(n, 2, figsize=(14, n * 3), facecolor=PALETTE["bg"])
    fig.suptitle("Signal Reconstruction: Normal vs Anomaly", color=PALETTE["text"], fontsize=15, y=1.01)

    for i in range(n):
        t = np.arange(X_normal.shape[1])

        for col, (original, recon, label, color) in enumerate([
            (X_normal[i, :, 0],  recon_normal[i, :, 0],  "Normal",  PALETTE["normal"]),
            (X_anomaly[i, :, 0], recon_anomaly[i, :, 0], "Anomaly", PALETTE["anomaly"]),
        ]):
            ax = axes[i, col]
            ax.set_facecolor(PALETTE["surface"])
            ax.plot(t, original, color=color,     lw=1.8, alpha=0.9, label="Original")
            ax.plot(t, recon,    color=PALETTE["accent"], lw=1.5, linestyle="--", alpha=0.85, label="Reconstructed")
            ax.fill_between(t, original, recon, alpha=0.15, color=PALETTE["accent"], label="Residual")

            mae = np.mean(np.abs(original - recon))
            ax.set_title(f"{label} — MAE: {mae:.4f}", color=PALETTE["text"], fontsize=11)
            ax.tick_params(colors=PALETTE["text"], labelsize=8)
            ax.spines[["top","right"]].set_visible(False)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

            if i == 0 and col == 0:
                ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"], fontsize=8)

    plt.tight_layout()
    _save(fig, "02_signal_reconstructions.png")
    return fig


# ── 3. Reconstruction error distribution ─────────────────────────────────────
def plot_error_distributions(normal_errors: np.ndarray, anomaly_errors: np.ndarray,
                              threshold: float) -> plt.Figure:
    """
    KDE plot of reconstruction errors for normal vs anomalous signals.
    The threshold line separates the two distributions.
    """
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    df = pd.DataFrame({
        "Reconstruction Error": np.concatenate([normal_errors, anomaly_errors]),
        "Class": ["Normal"] * len(normal_errors) + ["Anomaly"] * len(anomaly_errors)
    })

    sns.kdeplot(data=df, x="Reconstruction Error", hue="Class",
                fill=True, alpha=0.45, linewidth=2,
                palette={"Normal": PALETTE["normal"], "Anomaly": PALETTE["anomaly"]},
                ax=ax)

    ax.axvline(threshold, color=PALETTE["accent"], lw=2, linestyle="--", label=f"Threshold = {threshold:.4f}")
    ax.set_title("Reconstruction Error Distribution", color=PALETTE["text"], fontsize=14, pad=12)
    ax.set_xlabel("Mean Absolute Error", color=PALETTE["text"])
    ax.set_ylabel("Density", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.spines[["top","right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    handles = [
        mpatches.Patch(color=PALETTE["normal"],  label="Normal"),
        mpatches.Patch(color=PALETTE["anomaly"], label="Anomaly"),
        plt.Line2D([0], [0], color=PALETTE["accent"], lw=2, linestyle="--", label=f"Threshold ({threshold:.4f})"),
    ]
    ax.legend(handles=handles, facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])

    plt.tight_layout()
    _save(fig, "03_error_distributions.png")
    return fig


# ── 4. Confusion matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(normal_errors: np.ndarray, anomaly_errors: np.ndarray,
                           threshold: float) -> plt.Figure:
    """Seaborn heatmap confusion matrix with accuracy metrics."""
    y_true = np.array([0] * len(normal_errors) + [1] * len(anomaly_errors))
    y_pred = np.array(
        [0 if e <= threshold else 1 for e in normal_errors] +
        [0 if e <= threshold else 1 for e in anomaly_errors]
    )

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=PALETTE["bg"])

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Rates)"]
    ):
        ax.set_facecolor(PALETTE["surface"])
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"],
                    ax=ax, linewidths=0.5, linecolor="#333",
                    cbar_kws={"shrink": 0.8})
        ax.set_title(title, color=PALETTE["text"], fontsize=12, pad=10)
        ax.set_xlabel("Predicted", color=PALETTE["text"])
        ax.set_ylabel("Actual", color=PALETTE["text"])
        ax.tick_params(colors=PALETTE["text"])

    # Print metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / (tp + tn + fp + fn)

    metrics_text = (f"Accuracy: {accuracy:.3f}  |  Precision: {precision:.3f}  |  "
                    f"Recall: {recall:.3f}  |  F1: {f1:.3f}")
    fig.text(0.5, -0.02, metrics_text, ha="center", color=PALETTE["accent"], fontsize=11)

    plt.tight_layout()
    _save(fig, "04_confusion_matrix.png")
    return fig


# ── 5. ROC + Precision-Recall curves ─────────────────────────────────────────
def plot_roc_pr_curves(normal_errors: np.ndarray, anomaly_errors: np.ndarray) -> plt.Figure:
    """ROC and Precision-Recall curves with AUC scores."""
    y_true  = np.array([0] * len(normal_errors) + [1] * len(anomaly_errors))
    scores  = np.concatenate([normal_errors, anomaly_errors])

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc      = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=PALETTE["bg"])

    # ROC
    ax = axes[0]
    ax.set_facecolor(PALETTE["surface"])
    ax.plot(fpr, tpr, color=PALETTE["normal"], lw=2.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1], color="#555", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color=PALETTE["normal"])
    ax.set_title("ROC Curve", color=PALETTE["text"], fontsize=13)
    ax.set_xlabel("False Positive Rate", color=PALETTE["text"])
    ax.set_ylabel("True Positive Rate",  color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])
    ax.spines[["top","right"]].set_visible(False)

    # PR
    ax = axes[1]
    ax.set_facecolor(PALETTE["surface"])
    ax.plot(recall, precision, color=PALETTE["anomaly"], lw=2.5, label=f"AUC = {pr_auc:.3f}")
    ax.fill_between(recall, precision, alpha=0.1, color=PALETTE["anomaly"])
    ax.set_title("Precision-Recall Curve", color=PALETTE["text"], fontsize=13)
    ax.set_xlabel("Recall",    color=PALETTE["text"])
    ax.set_ylabel("Precision", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])
    ax.spines[["top","right"]].set_visible(False)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    plt.tight_layout()
    _save(fig, "05_roc_pr_curves.png")
    return fig


# ── 6. Latent space (PCA) ─────────────────────────────────────────────────────
def plot_latent_space(encoder, X_normal: np.ndarray, X_anomaly: np.ndarray,
                      max_samples: int = 500) -> plt.Figure:
    """
    Project the encoder's latent representations into 2D via PCA.
    Normal and anomalous signals should cluster separately in a well-trained model.
    """
    n = min(max_samples, len(X_normal), len(X_anomaly))
    latent_normal  = encoder.predict(X_normal[:n],  verbose=0)
    latent_anomaly = encoder.predict(X_anomaly[:n], verbose=0)

    latent_all = np.vstack([latent_normal, latent_anomaly])
    pca        = PCA(n_components=2)
    coords     = pca.fit_transform(latent_all)
    var_ratio  = pca.explained_variance_ratio_

    df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Class": ["Normal"] * n + ["Anomaly"] * n
    })

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Class",
                    palette={"Normal": PALETTE["normal"], "Anomaly": PALETTE["anomaly"]},
                    alpha=0.55, s=18, ax=ax, edgecolor="none")

    ax.set_title("Latent Space (PCA Projection)", color=PALETTE["text"], fontsize=14, pad=12)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% variance)", color=PALETTE["text"])
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% variance)", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.spines[["top","right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    legend = ax.get_legend()
    legend.get_frame().set_facecolor(PALETTE["surface"])
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])

    plt.tight_layout()
    _save(fig, "06_latent_space_pca.png")
    return fig


# ── 7. Error heatmap over time (time-series anomaly map) ─────────────────────
def plot_anomaly_timeline(model, X_test: np.ndarray, y_true: np.ndarray,
                          threshold: float, n_signals: int = 80) -> plt.Figure:
    """
    Colour-coded timeline showing reconstruction error per signal.
    Red = anomaly detected, blue = normal. Dashed line = threshold.
    """
    X_sub  = X_test[:n_signals]
    y_sub  = y_true[:n_signals]
    errors = np.mean(np.abs(X_sub - model.predict(X_sub, verbose=0)), axis=(1, 2))

    fig, ax = plt.subplots(figsize=(14, 4), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    colors = [PALETTE["anomaly"] if e > threshold else PALETTE["normal"] for e in errors]
    ax.bar(range(n_signals), errors, color=colors, width=0.85, alpha=0.85)
    ax.axhline(threshold, color=PALETTE["accent"], lw=1.5, linestyle="--",
               label=f"Threshold ({threshold:.4f})")

    ax.set_title("Anomaly Detection Timeline", color=PALETTE["text"], fontsize=14, pad=12)
    ax.set_xlabel("Signal Index", color=PALETTE["text"])
    ax.set_ylabel("Reconstruction Error (MAE)", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.spines[["top","right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    handles = [
        mpatches.Patch(color=PALETTE["normal"],  label="Predicted Normal"),
        mpatches.Patch(color=PALETTE["anomaly"], label="Predicted Anomaly"),
        plt.Line2D([0],[0], color=PALETTE["accent"], lw=2, linestyle="--", label="Threshold"),
    ]
    ax.legend(handles=handles, facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])

    plt.tight_layout()
    _save(fig, "07_anomaly_timeline.png")
    return fig
