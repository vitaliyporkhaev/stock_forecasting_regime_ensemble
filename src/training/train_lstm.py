import numpy as np
from src.models.lstm_model import LSTMModel


def create_sequences(X, y, seq_len=20):
    Xs, ys = [], []

    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])

    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train):
    X_seq, y_seq = create_sequences(X_train, y_train)

    model = LSTMModel(input_shape=(X_seq.shape[1], X_seq.shape[2]))
    model.fit(X_seq, y_seq, epochs=10)

    return model
