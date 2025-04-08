import yfinance as yf
import ta
import pandas as pd
import os

# Funzione per calcolare la percentuale di previsione
def calcola_punteggio(indicatori):
    punteggio = 0
    if indicatori["RSI (14)"] > 70:
        punteggio -= 8
    elif indicatori["RSI (14)"] < 30:
        punteggio += 8
    else:
        punteggio += 4

    if indicatori["MACD Line"] > indicatori["MACD Signal"]:
        punteggio += 8
    else:
        punteggio -= 6

    if indicatori["Stochastic %K"] > 80:
        punteggio -= 6
    elif indicatori["Stochastic %K"] < 20:
        punteggio += 6

    if indicatori["EMA (10)"] < indicatori["Stochastic %K"]:
        punteggio += 7

    if indicatori["CCI (14)"] > 0:
        punteggio += 6
    else:
        punteggio -= 4

    if indicatori["Williams %R"] > -20:
        punteggio -= 4
    else:
        punteggio += 4

    return round(((punteggio + 39) * 100) / 78, 2)

# Funzione per calcolare gli indicatori
def calcola_indicatori(ticker):
    data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
    data.dropna(inplace=True)

    close = data['Close'].squeeze()
    high = data['High'].squeeze()
    low = data['Low'].squeeze()

    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]
    macd_line = ta.trend.MACD(close=close).macd().iloc[-1]
    macd_signal = ta.trend.MACD(close=close).macd_signal().iloc[-1]
    stoch_k = ta.momentum.StochasticOscillator(high=high, low=low, close=close).stoch().iloc[-1]
    stoch_d = ta.momentum.StochasticOscillator(high=high, low=low, close=close).stoch_signal().iloc[-1]
    ema_10 = ta.trend.EMAIndicator(close=close, window=10).ema_indicator().iloc[-1]
    cci = ta.trend.CCIIndicator(high=high, low=low, close=close, window=14).cci().iloc[-1]
    williams_r = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r().iloc[-1]

    indicatori = {
        "RSI (14)": round(rsi, 2),
        "MACD Line": round(macd_line, 2),
        "MACD Signal": round(macd_signal, 2),
        "Stochastic %K": round(stoch_k, 2),
        "Stochastic %D": round(stoch_d, 2),
        "EMA (10)": round(ema_10, 2),
        "CCI (14)": round(cci, 2),
        "Williams %R": round(williams_r, 2),
    }

    return indicatori

# Lista degli asset
assets = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]

# Lista dei risultati da salvare
risultati = []

for ticker in assets:
    try:
        indicatori = calcola_indicatori(ticker)
        percentuale = calcola_punteggio(indicatori)
        risultati.append({
            "Asset": ticker,
            "Probabilità Crescita (%)": percentuale,
            **indicatori
        })
    except Exception as e:
        print(f"Errore per {ticker}: {e}")

# Salvataggio HTML
df = pd.DataFrame(risultati)
os.makedirs("results", exist_ok=True)
df.to_html("results/indicatori.html", index=False, border=1, classes="table table-striped")
