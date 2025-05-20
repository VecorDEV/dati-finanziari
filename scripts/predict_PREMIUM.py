import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings("ignore")

# Config
assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
threshold = 0.01  # +1%
n_past_days = 20
results = []

for symbol in assets:
    print(f"\n📊 Elaborazione {symbol}...")

    df = yf.download(symbol, period="5y", interval="1d", auto_adjust=True)
    df = df.dropna()

    # Indicatori tecnici
    df['RSI'] = RSIIndicator(close=df['Close']).rsi().values.flatten()
df['MACD'] = MACD(close=df['Close']).macd().values.flatten()
df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume().values.flatten()
    df = df.dropna()

    # Target: crescita percentuale > threshold nel giorno successivo
    df['Target'] = (df['Close'].shift(-1) > df['Close'] * (1 + threshold)).astype(int)
    df.dropna(inplace=True)

    # Feature scaling
    features = ['Close', 'RSI', 'MACD', 'OBV']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    target = df['Target'].values

    # Creazione sequenze
    X, y = [], []
    for i in range(n_past_days, len(scaled_data)):
        X.append(scaled_data[i - n_past_days:i])
        y.append(target[i])
    X, y = np.array(X), np.array(y)

    # Bilanciamento classi
    X_df = pd.DataFrame({'sequence': list(X), 'label': y})
    class_0 = X_df[X_df['label'] == 0]
    class_1 = X_df[X_df['label'] == 1]
    if len(class_1) > 0:
        class_0_downsampled = resample(class_0, replace=False, n_samples=len(class_1), random_state=42)
        X_balanced = pd.concat([class_1, class_0_downsampled])
    else:
        X_balanced = class_0
    X_balanced = X_balanced.sample(frac=1, random_state=42)
    X = np.stack(X_balanced['sequence'].values)
    y = X_balanced['label'].values

    # Split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Modello LSTM
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Addestramento
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)

    # Valutazione
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    last_prob = model.predict(X[-1:])[0][0] * 100

    print(f"✅ {symbol} → Probabilità crescita +1%: {last_prob:.2f}%")
    print(f"   - Accuratezza storica: {acc:.2f}%")
    print(f"   - Precisione: {prec:.2f}%")
    print(f"   - Recall: {rec:.2f}%")
    print(f"   - Confusion matrix: {conf_matrix}")

    results.append({
        "Asset": symbol,
        "Probabilità_Crescita_+1%": round(last_prob, 2),
        "Accuratezza_Storica": round(acc, 2),
        "Precisione": round(prec, 2),
        "Recall": round(rec, 2),
        "Confusion_Matrix": conf_matrix
    })

# Salva risultati
results_df = pd.DataFrame(results)
results_df.to_csv("risultati_premium.csv", index=False)
print("\n📁 Risultati salvati in 'risultati_premium.csv'")
