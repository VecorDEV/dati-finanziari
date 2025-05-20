import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Parametri
assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
prediction_threshold = 0.01  # 1% crescita

# Funzione di preprocessing
def preprocess_data(df):
    df['RSI'] = RSIIndicator(close=df['Close']).rsi().values.flatten()
    df['MACD'] = MACD(close=df['Close']).macd().values.flatten()
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume().values.flatten()
    df = df.dropna()
    return df

# Analisi per ciascun asset
results = []

for symbol in assets:
    print(f"\n📊 Elaborazione {symbol}...")

    df = yf.download(symbol, period="5y", interval="1d", auto_adjust=True)
    df = preprocess_data(df)

    # Target: cresce > +1% il giorno dopo
    df['Target'] = (df['Close'].shift(-1) > df['Close'] * (1 + prediction_threshold)).astype(int)
    df = df.dropna()

    X = df[['RSI', 'MACD', 'OBV']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba([X.iloc[-1]])[0][1] * 100

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, zero_division=0) * 100
    recall = recall_score(y_test, y_pred, zero_division=0) * 100
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"✅ {symbol} → Probabilità crescita +1%: {y_prob:.2f}%")
    print(f"   - Accuratezza storica: {accuracy:.2f}%")
    print(f"   - Precisione: {precision:.2f}%")
    print(f"   - Recall: {recall:.2f}%")
    print(f"   - Confusion matrix: {cm}")

    results.append({
        "Asset": symbol,
        "Probabilità_Crescita_+1%": y_prob,
        "Accuratezza_Storica": accuracy,
        "Precisione": precision,
        "Recall": recall
    })

# Mostra risultati tabellari
final_df = pd.DataFrame(results)
print("\n📈 Risultato finale (soglia +1%):\n")
print(final_df.to_string(index=False))
