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

# --- IMPATTO EVENTO ---
def analyze_impact(events_df, asset_series, days=[1, 3, 5, 7]):
    impact_rows = []
    for i in range(1, len(events_df)):
        row = events_df.iloc[i]
        prev = events_df.iloc[i - 1]
        event_date = pd.to_datetime(row["date"])
        
        # Trova la data più vicina nel dataset dell'asset
        closest_asset_date = asset_series.index.get_loc(event_date, method='nearest')
        closest_date = asset_series.index[closest_asset_date]
        
        # Verifica che la data dell'asset più vicina sia in un intervallo accettabile
        if abs((closest_date - event_date).days) > 30:
            continue  # Scarta se la differenza tra le date è troppo alta (ad esempio 30 giorni)
        
        direction = "up" if row["value"] > prev["value"] else "down"
        
        for d in days:
            future_date = closest_date + timedelta(days=d)
            if future_date in asset_series.index:
                change_pct = (asset_series[future_date] - asset_series[closest_date]) / asset_series[closest_date] * 100
                impact_rows.append({
                    "event_date": event_date,
                    "day_offset": d,
                    "change_pct": change_pct,
                    "event_value": row["value"],
                    "direction": direction
                })
    return pd.DataFrame(impact_rows)

# --- CALCOLO IMPACT SCORE ---
def calculate_impact_score(impact_df):
    if impact_df.empty:
        return 0.0
    abs_changes = impact_df["change_pct"].abs()
    avg_move = abs_changes.mean()
    std_dev = abs_changes.std()
    freq_strong_move = (abs_changes > 2).sum() / len(impact_df)
    score = (avg_move * 0.5) + (std_dev * 0.3) + (freq_strong_move * 100 * 0.2)
    return round(score, 2)

# --- ANALISI DIREZIONALE ---
def analyze_direction(impact_df):
    total = len(impact_df)
    if total == 0:
        return {"pos_pct": 0, "neg_pct": 0, "correlation": 0, "direction": "Neutral"}
    pos_pct = (impact_df["change_pct"] > 0).sum() / total * 100
    neg_pct = (impact_df["change_pct"] < 0).sum() / total * 100
    correlation = impact_df["event_value"].corr(impact_df["change_pct"])
    mean_change = impact_df["change_pct"].mean()
    if mean_change > 0.3:
        direction = "Positive"
    elif mean_change < -0.3:
        direction = "Negative"
    else:
        direction = "Neutral"
    return {
        "pos_pct": round(pos_pct, 2),
        "neg_pct": round(neg_pct, 2),
        "correlation": round(correlation, 2) if correlation is not None else 0,
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
