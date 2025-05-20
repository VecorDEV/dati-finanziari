import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator
from ta.trend import MACD

assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
sequence_length = 60
results = []

for symbol in assets:
    print(f"\n📊 Elaborazione {symbol}...")

    df = yf.download(symbol, start="2015-01-01", progress=False)

    if df.shape[0] < sequence_length + 1:
        print(f"⚠️ Dati insufficienti per {symbol}, saltato.")
        continue

    # Indicatori
    close_series = pd.Series(df['Close'].values.flatten(), index=df.index)
    df['RSI'] = RSIIndicator(close=close_series).rsi()
    macd = MACD(close=close_series)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    df['Volume_Change'] = df['Volume'].pct_change()

    df.dropna(inplace=True)

    features = ['Close', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'Volume_Change']
    data = df[features]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(1 if scaled_data[i][0] > scaled_data[i-1][0] else 0)

    X, y = np.array(X), np.array(y)

    if len(X) < 100:
        print(f"⚠️ Troppi pochi dati utili dopo il preprocessing per {symbol}, saltato.")
        continue

    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Modello
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Backtest: valutazione su test set
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    accuracy = (y_pred_class == y_test).mean()
    acc_percent = round(accuracy * 100, 2)

    # Previsione attuale
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    current_prediction = model.predict(last_sequence, verbose=0)[0][0]
    prob_growth = round(current_prediction * 100, 2)

    print(f"✅ {symbol}: Probabilità crescita attuale = {prob_growth}%, Accuratezza storica = {acc_percent}%")
    results.append({
        "Asset": symbol,
        "Probabilità_Crescita": prob_growth,
        "Accuratezza_Storica": acc_percent
    })

# Risultato finale
df_results = pd.DataFrame(results)
print("\n📈 Risultato finale:")
print(df_results.to_string(index=False))

# CSV opzionale
df_results.to_csv("backtest_risultati.csv", index=False)
