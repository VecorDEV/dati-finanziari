import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Lista di asset da analizzare
assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NRG", "NFLX", "BTCUSD"]

# Parametri modello
sequence_length = 60
results = []

for symbol in assets:
    print(f"\n📊 Elaborazione {symbol}...")

    # Scarica i dati
    df = yf.download(symbol, start="2015-01-01", progress=False)

    # Salta asset se i dati sono troppo pochi
    if df.shape[0] < sequence_length + 1:
        print(f"⚠️ Dati insufficienti per {symbol}, saltato.")
        continue

    # Indicatori tecnici
    close_series = pd.Series(df['Close'].values.flatten(), index=df.index)
    df['RSI'] = RSIIndicator(close=close_series).rsi()
    macd = MACD(close=close_series)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    df['Volume_Change'] = df['Volume'].pct_change()

    df.dropna(inplace=True)

    # Features per LSTM
    features = ['Close', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'Volume_Change']
    data = df[features]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Sequenze temporali
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(1 if scaled_data[i][0] > scaled_data[i-1][0] else 0)

    X, y = np.array(X), np.array(y)

    if len(X) < 100:
        print(f"⚠️ Troppi pochi dati utili dopo il preprocessing per {symbol}, saltato.")
        continue

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Costruzione e training modello
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

    # Calcolo probabilità
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence, verbose=0)[0][0]
    prob_growth = round(prediction * 100, 2)

    print(f"✅ {symbol}: Probabilità di crescita = {prob_growth}%")
    results.append({"Asset": symbol, "Probabilità_Crescita": prob_growth})

# Output finale tabellare
print("\n📈 Risultato finale:")
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Salva anche su CSV (opzionale)
df_results.to_csv("probabilita_crescita.csv", index=False)
