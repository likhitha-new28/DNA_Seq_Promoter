"""The convolutional neural network used for sequence classification."""

from __future__ import annotations


def build_cnn(sequence_length: int, number_of_classes: int = 3):
    """Build and compile a compact 1D CNN.

    TensorFlow is imported here so data utilities remain usable without loading it.
    """
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, 4), name="one_hot_dna"),
            tf.keras.layers.Conv1D(64, kernel_size=9, activation="relu", name="motif_filters"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(32, kernel_size=5, activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                number_of_classes, activation="softmax", name="class_probabilities"
            ),
        ],
        name="dna_regulatory_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def training_callbacks(model_path: str):
    """Return callbacks that avoid overfitting and retain the best validation model."""
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]
