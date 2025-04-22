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

# --- PARAMETRI ---
fred_api_key = sys.argv[1]
fred = Fred(api_key=fred_api_key)

SIGNIFICANT_MACRO_CHANGE = 0.3  # soglia percentuale per evento macro significativo
SIGNIFICANT_ASSET_REACTION = 1.5  # soglia reazione significativa su asset
WINDOW_DAYS = 10  # finestra temporale per reazione asset
MIN_OCCURRENCES = 20  # soglia per occorrenze significative

# --- FUNZIONI ---

def download_fred_series(series_id, years_back=20):
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.DateOffset(years=years_back)
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
    return yf.download(ticker, period="20y")["Close"]

# --- CALCOLO IMPATTO MACRO SUGLI ASSET ---
impact_results = []

for macro_name, series_id in FRED_SERIES.items():
    macro_df = download_fred_series(series_id)

    if macro_df.empty or "change_pct" not in macro_df.columns:
        continue

    # Prendi l'ultima variazione
    last_change_pct = macro_df["change_pct"].iloc[-1]
    latest_direction = "up" if last_change_pct >= SIGNIFICANT_MACRO_CHANGE else ("down" if last_change_pct <= -SIGNIFICANT_MACRO_CHANGE else "none")
    latest_macro_value = macro_df.iloc[-1]

    print(f"\nUltimo dato di {macro_name}: {latest_macro_value['value']} con variazione {round(last_change_pct, 2)}% ({latest_direction})")

    # Analizza entrambe le direzioni: up e down
    for direction in ["up", "down"]:
        for ticker, asset_name in ASSETS.items():
            asset_data = get_asset_data(ticker)
            if asset_data.empty:
                continue

            positive_reactions = 0
            negative_reactions = 0
            total_events = 0
            magnitudes = []

            for date, row in macro_df.iterrows():
                if row["value_change"] == direction:
                    event_date = date
                    if event_date not in asset_data.index:
                        nearest_idx = asset_data.index.get_indexer([event_date], method="nearest")[0]
                        event_date = asset_data.index[nearest_idx]

                    if event_date + pd.Timedelta(days=WINDOW_DAYS) > asset_data.index[-1]:
                        continue

                    start_price = float(asset_data.loc[event_date])
                    future_idx = asset_data.index.get_indexer([event_date + pd.Timedelta(days=WINDOW_DAYS)], method="nearest")[0]
                    end_price = float(asset_data.iloc[future_idx])

                    change_pct = (end_price - start_price) / start_price * 100

                    # Considera solo cambiamenti significativi (superiori alla soglia definita)
                    if abs(change_pct) < SIGNIFICANT_ASSET_REACTION:
                        continue

                    total_events += 1
                    magnitudes.append(change_pct)  # Aggiungi la magnitudine alla lista

                    if change_pct > 0:
                        positive_reactions += 1
                    else:
                        negative_reactions += 1

            if total_events >= MIN_OCCURRENCES:  # Filtra solo gli asset con sufficiente numero di eventi
                avg_magnitude = np.mean(magnitudes)  # Calcola la magnitudine media
                pos_pct = positive_reactions / total_events * 100
                neg_pct = negative_reactions / total_events * 100
                impact_results.append({
                    "Macro Factor": macro_name,
                    "Macro Direction": direction,
                    "Asset": ticker,
                    "Positive Impact %": round(pos_pct, 2),
                    "Negative Impact %": round(neg_pct, 2),
                    "Occurrences": total_events,
                    "Avg Magnitude %": round(avg_magnitude, 2)  # Aggiungi la magnitudine media
                })

# --- ESPORTA E MOSTRA ---
impact_df = pd.DataFrame(impact_results)

print("\n=== IMPACT SCORE COMPLETO ===")
print(impact_df)  # Mostra tutto il dataframe

impact_df.to_csv("impact_scores_recent.csv", index=False)
