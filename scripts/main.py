import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands

def fetch_all_features(symbol):
    # Scarica tutti i dati disponibili a intervallo giornaliero
    data = yf.download(symbol.upper(), interval="1d", auto_adjust=False)

    if data.empty:
        raise ValueError(f"Nessun dato disponibile per {symbol}.")

    data.dropna(inplace=True)

    # Calcolo indicatori
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    macd = MACD(data['Close'])
    data['MACD_line'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    stoch = StochasticOscillator(data['High'], data['Low'], data['Close'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    data['EMA10'] = EMAIndicator(data['Close'], window=10).ema_indicator()
    data['CCI'] = CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['WILLR'] = WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    bb = BollingerBands(data['Close'])
    data['BB_upper'] = bb.bollinger_hband()
    data['BB_lower'] = bb.bollinger_lband()
    data['BB_width'] = bb.bollinger_wband()

    # Medie mobili per confronto
    data['Volume_mean'] = data['Volume'].rolling(window=10).mean()
    data['BB_width_mean'] = data['BB_width'].rolling(window=10).mean()

    # Rimuove righe con NaN (iniziali degli indicatori)
    data.dropna(inplace=True)

    # Genera lista di feature binarie
    feature_matrix = []

    for _, row in data.iterrows():
        features = [
            int(row['Close'] > row['Open']),
            int(row['Volume'] > row['Volume_mean']),
            int(row['EMA10'] > row['Close']),
            int(row['RSI'] > 50),
            int(row['MACD_line'] > row['MACD_signal']),
            int(row['Stoch_K'] > row['Stoch_D']),
            int(row['CCI'] > 0),
            int(row['WILLR'] > -50),
            int(row['Close'] > row['BB_upper']),
            int(row['BB_width'] > row['BB_width_mean'])
        ]
        feature_matrix.append(features)

    return feature_matrix
    
    
symbol = "AAPL"
all_features = fetch_all_features(symbol)
print(all_features)




'''import numpy as np
from typing import List
import pennylane as qml

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normalize(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

def encode_qubit(x):
    theta = np.pi * x  # Assume x is normalized in [-1, 1]
    return theta

class QuantumSimModel:
    def __init__(self, n_features: int, hidden_size: int = 10,
                 lr: float = 0.01, reg: float = 0.001,
                 batch_size: int = 16, epochs: int = 200,
                 patience: int = 20, tol: float = 1e-4,
                 n_rotations: int = 3, window: int = 3):

        self.n = n_features
        self.k = n_rotations
        self.window = window
        self.hidden_size = hidden_size
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.tol = tol

        self.dev = qml.device("default.qubit", wires=n_features)

        self.thetas = np.random.uniform(0, 2*np.pi, (n_features, n_rotations))

        self.W1 = np.random.randn(hidden_size, n_features) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size) * 0.1
        self.b2 = 0.0

        self.m = {}
        self.v = {}
        for param in ['W1', 'b1', 'W2', 'b2']:
            self.m[param] = 0
            self.v[param] = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.iteration = 0

    def circuit(self, x, thetas):
        for i, val in enumerate(x):
            qml.RY(encode_qubit(val), wires=i)
        for i in range(self.n):
            for j in range(self.k):
                qml.RY(thetas[i, j], wires=i)
        for i in range(self.n - 1):
            qml.CNOT(wires=[i, i+1])
        return qml.probs(wires=range(self.n))

    def _simulate(self, x, thetas):
        qnode = qml.QNode(self.circuit, self.dev, interface='autograd')
        probs = qnode(x, thetas)
        index = 2 ** self.n - 1  # Index of |11...1>
        return probs[index]

    def _forward(self, p):
        z1 = self.W1 @ p + self.b1
        a1 = relu(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        out = sigmoid(z2)
        return out, a1

    def _loss(self, y_true, y_pred):
        return - (y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

    def _adam_update(self, param_name, grad):
        self.iteration += 1
        m = self.m[param_name]
        v = self.v[param_name]
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        m_hat = m / (1 - self.beta1 ** self.iteration)
        v_hat = v / (1 - self.beta2 ** self.iteration)
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.m[param_name] = m
        self.v[param_name] = v
        return update

    def fit(self, data: List[float], labels: List[int]):
        data = normalize(np.array(data))
        X, y = [], []
        for i in range(self.window, len(data)):
            X.append(data[i-self.window:i])
            y.append(labels[i])

        best_val_loss = float('inf')
        patience_counter = 0

        def quantum_forward(thetas_, x_):
            return self._simulate(x_, thetas_)

        grad_quantum_forward = qml.grad(quantum_forward, argnum=0)

        for epoch in range(self.epochs):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            total_loss = 0

            for start in range(0, len(X), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                grad_thetas = np.zeros_like(self.thetas)
                grad_W1 = np.zeros_like(self.W1)
                grad_b1 = np.zeros_like(self.b1)
                grad_W2 = np.zeros_like(self.W2)
                grad_b2 = 0.0

                for idx in batch_indices:
                    x_batch = X[idx]
                    y_batch = y[idx]

                    p = self._simulate(x_batch, self.thetas)
                    out, a1 = self._forward(p)
                    loss = self._loss(y_batch, out)
                    total_loss += loss

                    dL_dout = -(y_batch / (out + 1e-9)) + ((1 - y_batch) / (1 - out + 1e-9))
                    dout_dz2 = out * (1 - out)
                    dL_dz2 = dL_dout * dout_dz2

                    grad_W2 += dL_dz2 * a1
                    grad_b2 += dL_dz2

                    dz2_da1 = self.W2
                    da1_dz1 = (a1 > 0).astype(float)
                    dL_dz1 = dL_dz2 * dz2_da1 * da1_dz1

                    grad_W1 += np.outer(dL_dz1, p)
                    grad_b1 += dL_dz1

                    dL_dp = dL_dz1 @ self.W1
                    grad_p = grad_quantum_forward(self.thetas, x_batch)
                    grad_thetas += dL_dp[:, None] * grad_p

                grad_W1 += self.reg * self.W1
                grad_W2 += self.reg * self.W2
                grad_thetas += self.reg * self.thetas

                self.W1 -= self._adam_update('W1', grad_W1 / len(batch_indices))
                self.b1 -= self._adam_update('b1', grad_b1 / len(batch_indices))
                self.W2 -= self._adam_update('W2', grad_W2 / len(batch_indices))
                self.b2 -= self._adam_update('b2', grad_b2 / len(batch_indices))
                self.thetas -= self.lr * (grad_thetas / len(batch_indices))

            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

            if avg_loss + self.tol < best_val_loss:
                best_val_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def predict_proba(self, data: List[float]) -> float:
        data = normalize(np.array(data))
        x = data[-self.window:]
        p = self._simulate(x, self.thetas)
        out, _ = self._forward(p)
        return out

    def predict(self, data: List[float]) -> int:
        return int(self.predict_proba(data) >= 0.5)'''
