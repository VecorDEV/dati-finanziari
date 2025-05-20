import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Download dati
df = yf.download("AAPL", start="2015-01-01", end="2024-01-01")

# Calcolo indicatori con `ta`
df['RSI'] = RSIIndicator(close=df['Close']).rsi()
macd = MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_diff'] = macd.macd_diff()
df['Volume_Change'] = df['Volume'].pct_change()

# Drop NaN
df.dropna(inplace=True)

# Feature set
features = ['Close', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'Volume_Change']
data = df[features]

# Normalizzazione
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Costruzione sequenze per LSTM
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    # Target: chiusura successiva > chiusura attuale (1 = salita)
    y.append(1 if scaled_data[i][0] > scaled_data[i-1][0] else 0)

X, y = np.array(X), np.array(y)

# Divisione train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Costruzione modello LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Valutazione
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy su dati di test: {accuracy * 100:.2f}%")

# Ultima sequenza per previsione futura
last_sequence = scaled_data[-sequence_length:]
last_sequence = np.expand_dims(last_sequence, axis=0)
prediction = model.predict(last_sequence)[0][0]
prob_growth = prediction * 100
print(f"Probabilità stimata di crescita: {prob_growth:.2f}%")

# Grafico
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Prezzo di Chiusura')
plt.title(f"Probabilità stimata di crescita: {prob_growth:.2f}%")
plt.xlabel("Data")
plt.ylabel("Prezzo")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_segnali.png")
