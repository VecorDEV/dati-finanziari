import yfinance as yf
import ta
import pandas as pd
import os

# Scarica i dati
data = yf.download("TSLA", period="1y", interval="1d")
data.dropna(inplace=True)

# Calcola indicatori
data['rsi'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
data['macd'] = ta.trend.MACD(close=data['Close']).macd()
data['macd_signal'] = ta.trend.MACD(close=data['Close']).macd_signal()
data['sma_20'] = ta.trend.SMAIndicator(close=data['Close'], window=20).sma_indicator()

# Seleziona le ultime righe con gli indicatori
latest = data[['Close', 'rsi', 'macd', 'macd_signal', 'sma_20']].tail(30)  # ultime 30 righe

# Crea cartella results se non esiste
os.makedirs("results", exist_ok=True)

# Salva in HTML
html = latest.to_html(classes='table table-striped', border=0)
with open("results/indicatori.html", "w") as f:
    f.write(html)
