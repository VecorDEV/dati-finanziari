import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import sys
from fredapi import Fred

# --- CONFIGURAZIONE ---
ASSETS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta",
    "TSLA": "Tesla", "V": "Visa", "JPM": "JPMorgan", "JNJ": "Johnson & Johnson", "WMT": "Walmart",
    "NVDA": "NVIDIA", "PYPL": "PayPal", "DIS": "Disney", "NFLX": "Netflix"
}

FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "Unemployment": "UNRATE",
    "GDP": "GDP",
    "FedFunds": "FEDFUNDS"
}

fred_api_key = sys.argv[1]
fred = Fred(api_key=fred_api_key)

SIGNIFICANT_MACRO_CHANGE = 2.0  # soglia percentuale per eventi macro significativi
SIGNIFICANT_ASSET_REACTION = 1.0  # soglia percentuale per reazioni significative degli asset

def download_fred_series(series_id, years_back=10):
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.DateOffset(years=years_back)

    # Ottieni la Series e trasformala in DataFrame
    data = fred.get_series(series_id)
    data = data[data.index >= start_date]
    df = pd.DataFrame(data, columns=["value"])

    df["prev_value"] = df["value"].shift(1)
    df["change_pct"] = (df["value"] - df["prev_value"]) / df["prev_value"] * 100
    df["value_change"] = df["change_pct"].apply(
        lambda x: "up" if x >= SIGNIFICANT_MACRO_CHANGE else ("down" if x <= -SIGNIFICANT_MACRO_CHANGE else "none")
    )

    return df

def get_asset_data(ticker):
    return yf.download(ticker, period="10y")["Close"]

def get_nearest_date(dates, target_date):
    if len(dates) == 0:
        return None, None
    deltas = np.abs(dates - target_date)
    idx = int(np.argmin(deltas))
    return dates[idx], deltas[idx].days

# --- CALCOLO IMPACTO SUGLI ASSET IN BASE ALLA VARIAZIONE MACRO RECENTE ---
impact_results = []

for macro_name, series_id in FRED_SERIES.items():
    macro_df = download_fred_series(series_id)

    if macro_df.empty or "change_pct" not in macro_df.columns:
        continue

    # Prendi l'ultimo dato (il più recente)
    latest_macro_value = macro_df.iloc[-1]
    last_change_pct = macro_df["change_pct"].iloc[-1]
    latest_direction = "up" if last_change_pct >= SIGNIFICANT_MACRO_CHANGE else ("down" if last_change_pct <= -SIGNIFICANT_MACRO_CHANGE else "none")
    
    print(f"\nUltimo dato di {macro_name}: {latest_macro_value['value']} con variazione {round(last_change_pct, 2)}% ({latest_direction})")

    if latest_direction != "none":
        for ticker, asset_name in ASSETS.items():
            asset_data = get_asset_data(ticker)
            if asset_data.empty:
                continue

            positive_reactions = 0
            negative_reactions = 0
            total_events = 0

            for date, row in macro_df.iterrows():
                if row["value_change"] == latest_direction:
                    event_date = date
                    if event_date not in asset_data.index:
                        nearest_idx = asset_data.index.get_indexer([event_date], method="nearest")[0]
                        event_date = asset_data.index[nearest_idx]

                    if event_date not in asset_data.index or event_date + pd.Timedelta(days=5) > asset_data.index[-1]:
                        continue

                    start_price = float(asset_data.loc[event_date])
                    future_idx = asset_data.index.get_indexer([event_date + pd.Timedelta(days=5)], method="nearest")[0]
                    end_price = float(asset_data.iloc[future_idx])

                    change_pct = (end_price - start_price) / start_price * 100

                    if abs(change_pct) < SIGNIFICANT_ASSET_REACTION:
                        continue

                    total_events += 1
                    if change_pct > 0:
                        positive_reactions += 1
                    else:
                        negative_reactions += 1

            if total_events > 0:
                pos_pct = positive_reactions / total_events * 100
                neg_pct = negative_reactions / total_events * 100
            else:
                pos_pct = 0
                neg_pct = 0

            impact_results.append({
                "Macro Factor": macro_name,
                "Macro Direction": latest_direction,
                "Asset": ticker,
                "Positive Impact %": round(pos_pct, 2),
                "Negative Impact %": round(neg_pct, 2),
                "Occurrences": total_events
            })

# --- STAMPA RISULTATI FINALI ---
impact_df = pd.DataFrame(impact_results)
print("\n=== IMPACT SCORE COMPLETO (per aumento/diminuzione macro) ===")
print(impact_df.head(30))  # Mostra solo le prime 30 righe per leggibilità

# --- ESPORTA RISULTATI ---
impact_df.to_csv("impact_scores_recent.csv", index=False)
