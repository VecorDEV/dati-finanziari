import yfinance as yf
import ta
import pandas as pd
import os

from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands

# Crea la cartella results se non esiste
os.makedirs("results", exist_ok=True)

# Lista di asset da analizzare
assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "V", "JPM", "JNJ", "WMT",
        "NVDA", "PYPL", "DIS", "NFLX", "NIO", "NRG", "ADBE", "INTC", "CSCO", "PFE",
        "KO", "PEP", "MRK", "ABT", "XOM", "CVX", "T", "MCD", "NKE", "HD",
        "IBM", "CRM", "BMY", "ORCL", "ACN", "LLY", "QCOM", "HON", "COST", "SBUX",
        "CAT", "LOW", "MS", "GS", "AXP", "INTU", "AMGN", "GE", "FIS", "CVS",
        "DE", "BDX", "NOW", "SCHW", "LMT", "ADP", "C", "PLD", "NSC", "TMUS",
        "ITW", "FDX", "PNC", "SO", "APD", "ADI", "ICE", "ZTS", "TJX", "CL",
        "MMC", "EL", "GM", "CME", "EW", "AON", "D", "PSA", "AEP", "TROW", 
        "LNTH", "HE", "BTDR", "NAAS", "SCHL", "TGT", "SYK", "BKNG", "DUK", "USB"]

# Funzione per calcolare la percentuale in base agli indicatori
def calcola_punteggio(indicatori, close_price, bb_upper, bb_lower):
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

    if indicatori["EMA (10)"] < close_price:
        punteggio += 7

    if indicatori["CCI (14)"] > 0:
        punteggio += 6
    else:
        punteggio -= 4

    if indicatori["Williams %R"] > -20:
        punteggio -= 4
    else:
        punteggio += 4

    # Bollinger Bands
    if close_price > bb_upper:
        punteggio -= 5
    elif close_price < bb_lower:
        punteggio += 5

    return round(((punteggio + 44) * 100) / 88, 2)  # normalizzazione 0-100

# Analizza ogni asset
righe = []
for ticker in assets:
    try:
        print(f"Analizzando {ticker}...")
        
        # Scarica i dati storici per l'asset
        data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
        if data.empty:
            raise ValueError(f"Nessun dato disponibile per {ticker}.")
        
        data.dropna(inplace=True)

        close = data['Close'].squeeze()
        high = data['High'].squeeze()
        low = data['Low'].squeeze()

        # Indicatori tecnici
        rsi = RSIIndicator(close).rsi().iloc[-1]
        macd = MACD(close)
        macd_line = macd.macd().iloc[-1]
        macd_signal = macd.macd_signal().iloc[-1]
        stoch = StochasticOscillator(high, low, close)
        stoch_k = stoch.stoch().iloc[-1]
        stoch_d = stoch.stoch_signal().iloc[-1]
        ema_10 = EMAIndicator(close, window=10).ema_indicator().iloc[-1]
        cci = CCIIndicator(high, low, close).cci().iloc[-1]
        will_r = WilliamsRIndicator(high, low, close).williams_r().iloc[-1]

        bb = BollingerBands(close)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        bb_width = bb.bollinger_wband().iloc[-1]

        indicators = {
            "RSI (14)": round(rsi, 2),
            "MACD Line": round(macd_line, 2),
            "MACD Signal": round(macd_signal, 2),
            "Stochastic %K": round(stoch_k, 2),
            "Stochastic %D": round(stoch_d, 2),
            "EMA (10)": round(ema_10, 2),
            "CCI (14)": round(cci, 2),
            "Williams %R": round(will_r, 2),
            "BB Upper": round(bb_upper, 2),
            "BB Lower": round(bb_lower, 2),
            "BB Width": round(bb_width, 4),
        }

        percentuale = calcola_punteggio(indicators, close.iloc[-1], bb_upper, bb_lower)

        # Prepara la riga per la tabella
        riga = {"Asset": ticker, "Probabilità Crescita (%)": percentuale}
        riga.update(indicators)
        righe.append(riga)
        
    except Exception as e:
        # Gestione dell'errore per ciascun asset
        print(f"Errore durante l'analisi di {ticker}: {e}")

# Crea il DataFrame finale
df = pd.DataFrame(righe)

# Salva in HTML
df.to_html("results/indicatori.html", index=False, border=0, classes="table table-striped")

#CREA IL FILE DI CLASSIFICA
# Ordina i dati in base alla probabilità di crescita (decrescente)
df_sorted = df.sort_values(by="Probabilità Crescita (%)", ascending=False)

# Costruisci l'HTML personalizzato
html = "<html><head><title>Classifica dei Simboli</title></head><body>\n"
html += "<h1>Classifica dei Simboli in Base alla Probabilità di Crescita</h1>\n"
html += "<table border='1'><tr><th>Simbolo</th><th>Probabilità</th></tr>\n"

for _, row in df_sorted.iterrows():
    simbolo = row["Asset"]
    probabilita = f"{row['Probabilità Crescita (%)']:.2f}%"
    html += f"<tr><td>{simbolo}</td><td>{probabilita}</td></tr>\n"

html += "</table></body></html>"

# Scrivi l'HTML su file
with open("results/sortedByIndicators.html", "w", encoding="utf-8") as f:
    f.write(html)
        




'''import yfinance as yf
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
'''
