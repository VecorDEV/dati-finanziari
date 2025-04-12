import yfinance as yf
import pandas as pd

data = yf.download("AAPL", period="3mo", interval="1d", auto_adjust=True)
dati_storici = data.tail(90).copy().reset_index()
dati_storici['Date'] = dati_storici['Date'].dt.strftime('%Y-%m-%d')
dati_storici = dati_storici[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
print(dati_storici.head())
html = dati_storici.to_html(index=False, border=1)
print(html)
