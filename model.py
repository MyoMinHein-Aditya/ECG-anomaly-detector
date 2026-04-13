"""
ECG Anomaly Detector - Autoencoder Model
=========================================
A 1D Convolutional Autoencoder that learns to reconstruct normal ECG signals.
Anomalies are detected by measuring reconstruction error - abnormal signals
will have a higher error since the model never learned to reconstruct them.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(sequence_length: int = 140, latent_dim: int = 16) -> keras.Model:

    # ── Encoder ──────────────────────────────────────────────────────────────
    inputs = keras.Input(shape=(sequence_length, 1), name="ecg_input")

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu", name="enc_conv1")(inputs)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.MaxPooling1D(2, padding="same", name="enc_pool1")(x)
    x = layers.Dropout(0.1, name="enc_drop1")(x)

    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu", name="enc_conv2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.MaxPooling1D(2, padding="same", name="enc_pool2")(x)
    x = layers.Dropout(0.1, name="enc_drop2")(x)

    x = layers.Conv1D(16, kernel_size=5, padding="same", activation="relu", name="enc_conv3")(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)
    x = layers.MaxPooling1D(2, padding="same", name="enc_pool3")(x)

    encoded_shape = x.shape[1:]

    # Tight bottleneck — forces the model to compress aggressively
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # ── Decoder ──────────────────────────────────────────────────────────────
    x = layers.Dense(encoded_shape[0] * encoded_shape[1], activation="relu", name="dec_dense")(x)
    x = layers.Reshape(encoded_shape, name="reshape")(x)

    x = layers.Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", name="dec_conv1")(x)
    x = layers.BatchNormalization(name="dec_bn1")(x)
    x = layers.UpSampling1D(2, name="dec_up1")(x)

    x = layers.Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", name="dec_conv2")(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)
    x = layers.UpSampling1D(2, name="dec_up2")(x)

    x = layers.Conv1DTranspose(64, kernel_size=5, padding="same", activation="relu", name="dec_conv3")(x)
    x = layers.BatchNormalization(name="dec_bn3")(x)
    x = layers.UpSampling1D(2, name="dec_up3")(x)

    # Trim to exact input length
    x = layers.Lambda(lambda t: t[:, :sequence_length, :], name="trim")(x)

    outputs = layers.Conv1DTranspose(1, kernel_size=5, padding="same", activation="sigmoid", name="reconstruction")(x)

    # ── Compile ───────────────────────────────────────────────────────────────
    autoencoder = keras.Model(inputs, outputs, name="ECG_Autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae"
    )
    return autoencoder


def get_encoder(autoencoder: keras.Model) -> keras.Model:
    """Extract just the encoder portion for latent space visualisation."""
    return keras.Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer("latent").output,
        name="Encoder"
    )
