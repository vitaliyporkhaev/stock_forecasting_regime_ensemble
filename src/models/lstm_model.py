import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models


class LSTMModel:
    def __init__(self, input_shape):
        self.model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dense(1)
        ])

        self.model.compile(
            optimizer="adam",
            loss="mse"
        )

    def fit(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()
