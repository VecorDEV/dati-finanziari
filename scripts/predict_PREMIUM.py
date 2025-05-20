# filename: lstm_technical_indicators_aapl.py

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# === 1. Scarica dati storici AAPL ===
df = yf.download('AAPL', start='2010-01-01', end='2024-12-31')

# === 2. Calcolo indicatori tecnici ===
# Se non hai TA-Lib, installa con: pip install TA-Lib
import talib

df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['SMA20'] = df['Close'].rolling(window=20).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()

# Normalizza volumi
vol_scaler = MinMaxScaler()
df['Volume_norm'] = vol_scaler.fit_transform(df[['Volume']])

# Rimuovi NaN creati dagli indicatori
df.dropna(inplace=True)

# === 3. Prepara sequenze multivariate ===
features = ['Close', 'RSI', 'MACD', 'MACD_signal', 'SMA20', 'SMA50', 'Volume_norm']

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(scaled, columns=features, index=df.index)

# Target: 1 se il Close del giorno successivo è più alto, 0 altrimenti
df_scaled['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
df_scaled.dropna(inplace=True)

def create_sequences(data, feature_cols, target_col, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data.iloc[i-window:i][feature_cols].values)
        y.append(data.iloc[i][target_col])
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, features, 'Target', window=60)

# === 4. Split train/test e costruzione modello ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    verbose=2
)

# === 5. Predizioni probabilistiche ===
probs = model.predict(X_test).flatten()

# Aggiungi le predizioni al dataframe di test per allinearle alle date
test_idx = df_scaled.index[-len(probs):]
pred_df = pd.DataFrame({
    'Date': test_idx,
    'Close': df.loc[test_idx, 'Close'],
    'Prob_Growth': probs
}).set_index('Date')

# Stampa prime 10 probabilità
print("Prime 10 probabilità di crescita futura:")
print(pred_df['Prob_Growth'].head(10).apply(lambda p: f"{p:.2%}"))

# === 6. Visualizzazione segnali su grafico ===
threshold = 0.6
buy_signals = pred_df['Prob_Growth'] > threshold

plt.figure(figsize=(14, 6))
plt.plot(pred_df.index, pred_df['Close'], label='Close Price')
plt.scatter(
    pred_df.index[buy_signals],
    pred_df['Close'][buy_signals],
    marker='^', s=80, label=f'Buy (Prob > {threshold})'
)
plt.title('AAPL Close Price & Buy Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()
