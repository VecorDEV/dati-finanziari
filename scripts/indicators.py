import yfinance as yf
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.momentum import StochasticOscillator, WilliamsRIndicator
import pandas as pd
from datetime import datetime

# Funzione per calcolare la percentuale di previsione dell'andamento
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

# Funzione per calcolare gli indicatori tecnici
def calcola_indicatori(ticker):
    data = yf.download(ticker, period="1mo", interval="1d")  # Dati degli ultimi 30 giorni

    # Calcolo degli indicatori
    rsi = RSIIndicator(data['Close'], window=14).rsi().iloc[-1]  # Ultimo valore RSI
    macd = MACD(data['Close']).macd().iloc[-1]  # Ultimo valore MACD
    macd_signal = MACD(data['Close']).macd_signal().iloc[-1]  # Ultimo valore Signal MACD
    stochastic_k = StochasticOscillator(data['High'], data['Low'], data['Close'], window=14).stoch().iloc[-1]  # Ultimo valore Stochastic %K
    ema = EMAIndicator(data['Close'], window=10).ema_indicator().iloc[-1]  # Ultimo valore EMA (10)
    cci = CCIIndicator(data['High'], data['Low'], data['Close'], window=14).cci().iloc[-1]  # Ultimo valore CCI
    williams_r = WilliamsRIndicator(data['High'], data['Low'], data['Close'], window=14).williams_r().iloc[-1]  # Ultimo valore Williams %R

    # Creiamo un dizionario con i valori degli indicatori
    indicatori = {
        "RSI (14)": rsi,
        "MACD Line": macd,
        "MACD Signal": macd_signal,
        "Stochastic %K": stochastic_k,
        "EMA (10)": ema,
        "CCI (14)": cci,
        "Williams %R": williams_r
    }
    
    return indicatori

# Funzione per creare il file HTML con il nome dell'asset e la percentuale
def crea_file_html(nome_asset, percentuale):
    # Definisco il contenuto HTML
    html_content = f"""
    <html>
    <head>
        <title>Previsione Andamento {nome_asset}</title>
    </head>
    <body>
        <h1>Previsione Andamento dell'Asset: {nome_asset}</h1>
        <p>Percentuale di probabilità di crescita nel breve periodo: {percentuale}%</p>
    </body>
    </html>
    """

    # Salvo il contenuto HTML in un file nella cartella results
    with open("results/indicatori.html", "w") as file:
        file.write(html_content)

# Esempio di esecuzione per un asset
asset = "AAPL"  # Puoi cambiare con qualsiasi asset
indicatori = calcola_indicatori(asset)
percentuale = calcola_punteggio(indicatori)

# Creazione del file HTML con il nome dell'asset e la percentuale calcolata
crea_file_html(asset, round(percentuale, 2))
