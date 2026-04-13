# ECG Anomaly Detector — Autoencoder (TensorFlow + Seaborn)

Detect abnormal heartbeat patterns in ECG signals using an unsupervised 1D Convolutional Autoencoder.

---

## How it works

The core idea is simple and powerful:

1. **Train only on normal ECGs** — the autoencoder learns to reconstruct healthy heartbeats
2. **Measure reconstruction error** — when shown an anomalous ECG, the model reconstructs it poorly (high MAE)
3. **Threshold the error** — signals above the 95th percentile of normal errors are flagged as anomalies

No labels needed during training. This is **unsupervised anomaly detection**.

---

## Project structure

```
ecg_anomaly_detector/
│
├── data/
    ├── own dataset
├── model.py             # 1D Conv Autoencoder architecture (TensorFlow/Keras)
├── data_utils.py        # Data loading, preprocessing, threshold computation
├── visualisations.py    # 7 Seaborn/Matplotlib plots
├── train.py             # Main training script
├── predict.py           # Inference on new signals
└── Requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Download the ECG5000 dataset
The ECG5000 dataset from the UCR Time Series Archive is the standard benchmark.

Download from: http://www.timeseriesclassification.com/description.php?Dataset=ECG5000

Place `ECG5000_TRAIN.txt` in a `data/` folder and rename it `ecg5000_train.csv`.

> **Without the dataset**, the project automatically generates synthetic ECG-like signals so you can run it immediately.

### 3. Train
```bash
python train.py
```

This will:
- Load/generate ECG data
- Train the autoencoder for up to 50 epochs (with early stopping)
- Save the model to `outputs/ecg_autoencoder.keras`
- Generate 7 visualisation plots in `outputs/`

### 4. Predict on new signals
```bash
# With your own CSV (one signal per row, 140 columns):
python predict.py --input my_ecg_data.csv --threshold 0.05

# With plots for each signal:
python predict.py --input my_ecg_data.csv --plot

# Quick demo with synthetic signals:
python predict.py
```

---

## Model architecture

```
Input (140, 1)
    │
    ▼  Encoder
Conv1D(32, k=7) → BN → MaxPool(2)
Conv1D(16, k=7) → BN → MaxPool(2)
Conv1D( 8, k=7)
Flatten → Dense(32)        ← latent bottleneck
    │
    ▼  Decoder
Dense → Reshape
Conv1DTranspose(8,  k=7) → BN → UpSampling(2)
Conv1DTranspose(16, k=7) → BN → UpSampling(2)
Conv1DTranspose(32, k=7)
Conv1DTranspose(1,  k=7)   ← reconstruction
    │
Output (140, 1)
```

**Loss**: Mean Absolute Error (MAE) — more robust to outliers than MSE  
**Optimiser**: Adam (lr=1e-3 with ReduceLROnPlateau)

---

## Visualisations generated

| File | Description |
|------|-------------|
| `01_training_history.png` | Train vs validation MAE loss curves |
| `02_signal_reconstructions.png` | Original vs reconstructed for normal & anomalous ECGs |
| `03_error_distributions.png` | KDE of reconstruction errors with threshold line |
| `04_confusion_matrix.png` | Confusion matrix (counts + rates) + metrics |
| `05_roc_pr_curves.png` | ROC curve and Precision-Recall curve with AUC |
| `06_latent_space_pca.png` | PCA projection of the encoder's latent space |
| `07_anomaly_timeline.png` | Bar chart of errors over time, coloured by prediction |

---

## Key concepts

| Concept | What it teaches you |
|---------|---------------------|
| Autoencoder | Unsupervised representation learning |
| Reconstruction error | Threshold-based anomaly scoring |
| Conv1D | 1D convolutions for sequence data |
| Latent space | Compressed signal representation |
| Imbalanced evaluation | Why accuracy alone is misleading — use F1, AUC |
| PCA visualisation | Dimensionality reduction for interpretability |

---

## Extending the project

- **Use the real ECG5000 dataset** for benchmark-quality results
- **Try an LSTM Autoencoder** — replace Conv1D layers with LSTM layers
- **Tune the threshold** — experiment with different percentiles and see the precision/recall tradeoff
- **Add a Variational Autoencoder (VAE)** — learn a probabilistic latent space
- **Deploy with FastAPI** — wrap `predict.py` in an API endpoint
