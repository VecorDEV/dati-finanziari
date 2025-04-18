import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from fredapi import Fred

# --- CONFIGURAZIONE ---
ASSETS = {
    "BTC-USD": "Bitcoin",
    "AAPL": "Apple",
    "EURUSD=X": "Euro/Dollaro"
}

FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "Unemployment": "UNRATE",
    "GDP": "GDP",
    "FedFunds": "FEDFUNDS"
}

API_KEY = "586442cd31253d8596bdc4c2a28fdffe"  # <-- Inserisci qui la tua chiave
fred = Fred(api_key=API_KEY)

def download_fred_series(series_id, years_back=5):
    # Calcola la data di inizio per i dati
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.DateOffset(years=years_back)
    
    # Ottieni i dati dalla serie FRED
    data = fred.get_series(series_id)
    
    # Filtra i dati in base alla data di inizio
    data = data[data.index >= start_date]
    
    print(f"Dati scaricati per {series_id}: {len(data)} righe")
    
    # Rendi i dati un DataFrame con colonne 'date' e 'value'
    data = data.reset_index()
    data.columns = ["date", "value"]
    
    return data

# --- SCARICA DATI ASSET ---
def get_asset_data(ticker):
    return yf.download(ticker, period="3y")["Close"]


def get_nearest_date(index, target_date):
    """Ritorna la date dell'index più vicina a target_date e la differenza in giorni."""
    deltas = (index - target_date).days
    idx = int(np.argmin(np.abs(deltas)))
    nearest = index[idx]
    return nearest, abs((nearest - target_date).days)

def analyze_impact(events_df, asset_series, days=[1, 3, 5, 7], tol_days=3):
    impact_rows = []
    for i in range(1, len(events_df)):
        row  = events_df.iloc[i]
        prev = events_df.iloc[i-1]
        event_date = pd.to_datetime(row["date"])

        # 1) Trova il trading day più vicino all'evento
        start_date, diff_start = get_nearest_date(asset_series.index, event_date)
        if diff_start > tol_days:
            continue  # scarta se la differenza è > tol_days

        direction = "up" if row["value"] > prev["value"] else "down"

        # 2) Per ogni offset, trova il trading day più vicino
        for d in days:
            target = event_date + timedelta(days=d)
            end_date, diff_end = get_nearest_date(asset_series.index, target)
            if diff_end > tol_days:
                continue  # scarta se fuori tolleranza

            # 3) Calcola la variazione % tra start_date e end_date
            change_pct = (asset_series.loc[end_date]
                          - asset_series.loc[start_date]) / asset_series.loc[start_date] * 100

            impact_rows.append({
                "event_date":   event_date,
                "day_offset":   d,
                "change_pct":   change_pct,
                "event_value":  row["value"],
                "direction":    direction
            })

    return pd.DataFrame(impact_rows)

# --- CALCOLO IMPACT SCORE ---
def calculate_impact_score(impact_df):
    if impact_df.empty:
        return 0.0

    # Assicuriamoci che change_pct sia floating
    changes = impact_df["change_pct"].astype(float).abs()

    # Calcolo esplicito con cast a float
    avg_move     = float(changes.mean())
    std_dev      = float(changes.std())
    freq_strong  = float((changes > 2).sum()) / len(changes)

    score = (avg_move * 0.5) + (std_dev * 0.3) + (freq_strong * 100 * 0.2)
    return round(score, 2)

# --- ANALISI DIREZIONALE ---
def analyze_direction(impact_df):
    # Se non ci sono dati, ritorna neutro
    if impact_df.empty:
        return {"pos_pct": 0.0, "neg_pct": 0.0, "correlation": 0.0, "direction": "Neutral"}

    # Forziamo i tipi numerici
    changes = impact_df["change_pct"].astype(float)
    values  = impact_df["event_value"].astype(float)
    total   = len(changes)

    # Calcolo percentuali
    pos_pct = (changes > 0).sum() / total * 100.0
    neg_pct = (changes < 0).sum() / total * 100.0

    # Correlazione
    corr = float(values.corr(changes))

    # Direzione media
    mean_change = float(changes.mean())
    if mean_change > 0.3:
        direction = "Positive"
    elif mean_change < -0.3:
        direction = "Negative"
    else:
        direction = "Neutral"

    return {
        "pos_pct": round(pos_pct, 2),
        "neg_pct": round(neg_pct, 2),
        "correlation": round(corr, 2),
        "direction": direction
    }

# --- GENERA SEGNALE BUY/SELL ---
def generate_signal(score, direction, pos_pct, neg_pct):
    if score > 25 and direction == "Positive" and pos_pct > 60:
        return "BUY"
    elif score > 25 and direction == "Negative" and neg_pct > 60:
        return "SELL"
    else:
        return "NEUTRAL"

# --- MAIN ---
impact_summary = []

for ticker, asset_name in ASSETS.items():
    asset_data = get_asset_data(ticker)
    for event_name, fred_series_id in FRED_SERIES.items():
        events_df = download_fred_series(fred_series_id)
        events_df["date"] = pd.to_datetime(events_df["date"])
        impact_df = analyze_impact(events_df, asset_data)
        score = calculate_impact_score(impact_df)
        directionals = analyze_direction(impact_df)
        signal = generate_signal(score, directionals["direction"], directionals["pos_pct"], directionals["neg_pct"])
        impact_summary.append({
            "Event": event_name,
            "Asset": asset_name,
            "Ticker": ticker,
            "Impact Score": score,
            "Positive %": directionals["pos_pct"],
            "Negative %": directionals["neg_pct"],
            "Macro Corr.": directionals["correlation"],
            "Directional Impact": directionals["direction"],
            "Signal": signal
        })

# --- ESPORTA RISULTATO ---
summary_df = pd.DataFrame(impact_summary)
summary_df.to_csv("impact_scores_all.csv", index=False)
