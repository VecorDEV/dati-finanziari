import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from fredapi import Fred

# --- CONFIG ---
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

API_KEY = "586442cd31253d8596bdc4c2a28fdffe"
fred = Fred(api_key=API_KEY)

def download_fred_series(series_id):
    data = fred.get_series(series_id)
    data = data.reset_index()
    data.columns = ["date", "value"]
    return data

def get_asset_data(ticker):
    return yf.download(ticker, period="3y")["Close"]

def analyze_impact(events_df, asset_series, days=[1, 3, 5, 7]):
    impact_rows = []
    for i in range(1, len(events_df)):
        row = events_df.iloc[i]
        prev = events_df.iloc[i - 1]
        raw_date = pd.to_datetime(row["date"])

        idx = asset_series.index.get_indexer([raw_date], method='nearest')
        if idx[0] == -1:
            continue
        event_date = asset_series.index[idx[0]]

        direction = "up" if row["value"] > prev["value"] else "down"
        for d in days:
            future_date = event_date + timedelta(days=d)
            if future_date > asset_series.index[-1]:
                continue
            future_idx = asset_series.index.get_indexer([future_date], method='nearest')
            if future_idx[0] == -1:
                continue
            future_date = asset_series.index[future_idx[0]]
            change_pct = (asset_series[future_date] - asset_series[event_date]) / asset_series[event_date] * 100
            impact_rows.append({
                "event_date": event_date,
                "day_offset": d,
                "change_pct": change_pct,
                "event_value": row["value"],
                "direction": direction
            })
    return pd.DataFrame(impact_rows)

def calculate_impact_score(impact_df):
    if impact_df.empty:
        return 0.0
    abs_changes = impact_df["change_pct"].abs()
    avg_move = abs_changes.mean()
    std_dev = abs_changes.std()
    freq_strong_move = (abs_changes > 2).sum() / len(impact_df)
    score = (avg_move * 0.5) + (std_dev * 0.3) + (freq_strong_move * 100 * 0.2)
    return round(score, 2)

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
    print(f"Analisi {asset_name} ({ticker})")
    asset_data = get_asset_data(ticker)
    for event_name, fred_series_id in FRED_SERIES.items():
        print(f"  Evento: {event_name}")
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

summary_df = pd.DataFrame(impact_summary)
summary_df.to_csv("impact_scores_all.csv", index=False)
print("Creato il file impact_scores_all.csv")
