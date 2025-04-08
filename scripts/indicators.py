import yfinance as yf
import ta
import pandas as pd
import os

# Scarica dati (minimo 30-60 giorni per stabilità degli indicatori)
data = yf.download("TSLA", period="3mo", interval="1d", auto_adjust=True)
data.dropna(inplace=True)

close = data['Close'].squeeze()
high = data['High'].squeeze()
low = data['Low'].squeeze()

# Calcola gli indicatori principali
rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]
macd_line = ta.trend.MACD(close=close).macd().iloc[-1]
macd_signal = ta.trend.MACD(close=close).macd_signal().iloc[-1]
stoch_k = ta.momentum.StochasticOscillator(high=high, low=low, close=close).stoch().iloc[-1]
stoch_d = ta.momentum.StochasticOscillator(high=high, low=low, close=close).stoch_signal().iloc[-1]
ema_10 = ta.trend.EMAIndicator(close=close, window=10).ema_indicator().iloc[-1]
cci = ta.trend.CCIIndicator(high=high, low=low, close=close, window=14).cci().iloc[-1]
williams_r = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r().iloc[-1]

# Compila dizionario con i valori finali
indicators = {
    "RSI (14)": round(rsi, 2),
    "MACD Line": round(macd_line, 2),
    "MACD Signal": round(macd_signal, 2),
    "Stochastic %K": round(stoch_k, 2),
    "Stochastic %D": round(stoch_d, 2),
    "EMA (10)": round(ema_10, 2),
    "CCI (14)": round(cci, 2),
    "Williams %R": round(williams_r, 2),
}

# Crea tabella HTML
df = pd.DataFrame(indicators.items(), columns=["Indicatore", "Valore"])
os.makedirs("results", exist_ok=True)
df.to_html("results/indicatori.html", index=False, border=0, classes="table table-striped")
