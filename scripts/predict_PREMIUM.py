import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator
from ta.trend import MACD

assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
sequence_length = 60
future_days = 5
target_threshold = 0.02  # +2%
results = []

for symbol in assets:
    print(f"\n📊 Elaborazione {symbol}...")

    df = yf.download(symbol, start="2015-01-01", progress=False)

    if df.shape[0] < sequence_length + future_days:
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

    # Target: crescita ≥ +2% nei prossimi 5 giorni
    df['Target'] = (df['Close'].shift(-future_days) / df['Close']) - 1
    df['Target'] = df['Target'].apply(lambda x: 1 if x >= target_threshold else 0)

    df.dropna(inplace=True)

    features = ['Close', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'Volume_Change']
    data = df[features]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - future_days):
        X.append(scaled_data[i-sequence_length:i])
        y.append(df['Target'].iloc[i + future_days])

    X, y = np.array(X), np.array(y)

    if len(X) < 100:
        print(f"⚠️ Troppi pochi dati utili dopo il preprocessing per {symbol}, saltato.")
        continue

    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Modello LSTM
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Valutazione
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class, zero_division=0)

    acc_percent = round(acc * 100, 2)
    precision_percent = round(precision * 100, 2)

    # Previsione attuale
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    current_prediction = model.predict(last_sequence, verbose=0)[0][0]
    prob_growth = round(current_prediction * 100, 2)

    print(f"✅ {symbol}: Probabilità crescita +2% = {prob_growth}%, Acc = {acc_percent}%, Precisione = {precision_percent}%")

    results.append({
        "Asset": symbol,
        "Probabilità_Crescita_+2%": prob_growth,
        "Accuratezza_Storica": acc_percent,
        "Precisione_Crescita": precision_percent
    })

# Risultato finale
df_results = pd.DataFrame(results)
print("\n📈 Risultato finale (previsione + backtest):")
print(df_results.to_string(index=False))
df_results.to_csv("risultati_backtest_5d.csv", index=False)
