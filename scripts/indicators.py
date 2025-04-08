import yfinance as yf
import ta
import pandas as pd
import os

# Funzione per calcolare il punteggio di previsione
def calcola_punteggio(indicatori):
    punteggio = 0
    
    # RSI: se RSI > 70 (overbought), segnale negativo; se RSI < 30 (oversold), segnale positivo
    if indicatori["RSI (14)"] > 70:
        punteggio -= 8  # ipercomprato
    elif indicatori["RSI (14)"] < 30:
        punteggio += 8  # ipervenduto
    else:
        punteggio += 4  # neutro

    # MACD: crossover positivo MACD > Signal -> segnale positivo
    if indicatori["MACD Line"] > indicatori["MACD Signal"]:
        punteggio += 8  # tendenzialmente rialzista
    else:
        punteggio -= 6  # tendenzialmente ribassista

    # Stochastic: Stocastico sopra 80 (ipercomprato) o sotto 20 (ipervenduto)
    if indicatori["Stochastic %K"] > 80:
        punteggio -= 6  # ipercomprato
    elif indicatori["Stochastic %K"] < 20:
        punteggio += 6  # ipervenduto

    # EMA (10): prezzo sopra EMA -> segnale positivo
    if indicatori["EMA (10)"] < indicatori["Stochastic %K"]:
        punteggio += 7  # prezzo sopra la EMA (positiva)

    # CCI: valori positivi indicano un trend rialzista
    if indicatori["CCI (14)"] > 0:
        punteggio += 6
    else:
        punteggio -= 4

    # Williams %R: se sopra -20, segnale di ipercomprato
    if indicatori["Williams %R"] > -20:
        punteggio -= 4  # condizione di ipercomprato
    else:
        punteggio += 4  # condizione di ipervenduto

    # Normalizzare il punteggio tra 0 e 100
    punteggio_normalizzato = ((punteggio + 39) * 100) / 78  # valori tra 0 e 100

    return punteggio_normalizzato

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

# Calcola la previsione basata sugli indicatori
percentuale = calcola_punteggio(indicators)

# Crea tabella HTML
df = pd.DataFrame(indicators.items(), columns=["Indicatore", "Valore"])

# Aggiungi la previsione al contenuto HTML
os.makedirs("results", exist_ok=True)
html_content = f"""
<html>
<head>
    <title>Indicatori Tecnici e Previsione</title>
</head>
<body>
    <h1>Previsione Andamento dell'Asset: TSLA</h1>
    <p>Percentuale di probabilità di crescita nel breve periodo: {round(percentuale, 2)}%</p>
    <h2>Indicatori Tecnici</h2>
    {df.to_html(index=False, border=0, classes="table table-striped")}
</body>
</html>
"""

# Salva il contenuto HTML in un file
with open("results/indicatori.html", "w") as file:
    file.write(html_content)

print(f"Il file HTML con i dati e la previsione è stato salvato come 'results/indicatori.html'.")
