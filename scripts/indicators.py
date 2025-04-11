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
assets = ["AAPL", "MSFT"]

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


def aggiorna_file_html(ticker, percentuale, indicators, storico_df):
    file_path = f"results/{ticker}_RESULT.html"

    if not os.path.exists(file_path):
        print(f"File {file_path} non trovato, saltato.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        contenuto = f.read()

    # Crea tabella indicatori tecnici
    tabella_indicatori = pd.DataFrame(indicators.items(), columns=["Indicatore", "Valore"]).to_html(index=False, border=0)

    # Crea tabella dei dati storici (ultimi 90 giorni)
    storico_html = storico_df.tail(90).to_html(index=False, border=0)

    # Costruisci il nuovo blocco HTML
    nuovo_blocco = f"""
<!-- INDICATORI_INIZIO -->
<h2>Indicatori Tecnici</h2>
<p>Probabilità di crescita: <strong>{percentuale}%</strong></p>
{tabella_indicatori}

<h2>Dati Storici (ultimi 90 giorni)</h2>
{storico_html}
<!-- INDICATORI_FINE -->
"""

    # Se il blocco esiste già, sostituiscilo
    if "<!-- INDICATORI_INIZIO -->" in contenuto and "<!-- INDICATORI_FINE -->" in contenuto:
        contenuto = re.sub(
            r'<!-- INDICATORI_INIZIO -->.*?<!-- INDICATORI_FINE -->',
            nuovo_blocco,
            contenuto,
            flags=re.DOTALL
        )
    else:
        # Inserisci prima del </body> se non esiste ancora
        contenuto = contenuto.replace("</body>", f"{nuovo_blocco}</body>")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(contenuto)

    print(f"{ticker} aggiornato con indicatori tecnici e dati storici.")


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

        aggiorna_file_html(ticker, percentuale, indicators, data)
        
    except Exception as e:
        # Gestione dell'errore per ciascun asset
        print(f"Errore durante l'analisi di {ticker}: {e}")

# Crea il DataFrame finale
df = pd.DataFrame(righe)

# Salva in HTML
df.to_html("results/indicatori.html", index=False, border=0, classes="table table-striped")
