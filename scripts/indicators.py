import yfinance as yf
import ta
import pandas as pd
import os

# Scarica dati
data = yf.download("TSLA", period="1y", interval="1d")
data.dropna(inplace=True)

# Assicurati che la colonna Close sia 1D
close_prices = data['Close'].squeeze()

# Calcola indicatori
data['rsi'] = ta.momentum.RSIIndicator(close=close_prices).rsi()
macd = ta.trend.MACD(close=close_prices)
data['macd'] = macd.macd()
data['macd_signal'] = macd.macd_signal()
data['sma_20'] = ta.trend.SMAIndicator(close=close_prices, window=20).sma_indicator()

# Crea cartella results se non esiste
os.makedirs("results", exist_ok=True)

# Salva HTML con ultimi 30 giorni
latest = data[['Close', 'rsi', 'macd', 'macd_signal', 'sma_20']].tail(30)
html = latest.to_html(classes='table table-striped', border=0)
with open("results/indicatori.html", "w") as f:
    f.write(html)
