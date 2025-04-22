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

SIGNIFICANT_MACRO_CHANGE = 1.5
SIGNIFICANT_ASSET_REACTION = 1.0
REACTION_WINDOW_DAYS = 15
MIN_OCCURRENCES = 30

def download_fred_series(series_id, years_back=10):
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
    return yf.download(ticker, period="10y")["Close"]

impact_results = []

for macro_name, series_id in FRED_SERIES.items():
    macro_df = download_fred_series(series_id)

    if macro_df.empty or "change_pct" not in macro_df.columns:
        continue

    for direction in ["up", "down"]:
        filtered_macro_df = macro_df[macro_df["value_change"] == direction]

        if filtered_macro_df.empty:
            continue

        print(f"\nAnalisi {macro_name} ({direction}) - Eventi: {len(filtered_macro_df)}")

        for ticker, asset_name in ASSETS.items():
            asset_data = get_asset_data(ticker)
            if asset_data.empty:
                continue

            pos_changes = []
            neg_changes = []
            total_events = 0

            for date, row in filtered_macro_df.iterrows():
                event_date = date
                if event_date not in asset_data.index:
                    nearest_idx = asset_data.index.get_indexer([event_date], method="nearest")[0]
                    event_date = asset_data.index[nearest_idx]

                if event_date not in asset_data.index or event_date + pd.Timedelta(days=REACTION_WINDOW_DAYS) > asset_data.index[-1]:
                    continue

                start_price = float(asset_data.loc[event_date])
                future_idx = asset_data.index.get_indexer([event_date + pd.Timedelta(days=REACTION_WINDOW_DAYS)], method="nearest")[0]
                end_price = float(asset_data.iloc[future_idx])

                change_pct = (end_price - start_price) / start_price * 100

                if abs(change_pct) < SIGNIFICANT_ASSET_REACTION:
                    continue

                total_events += 1
                if change_pct > 0:
                    pos_changes.append(change_pct)
                else:
                    neg_changes.append(change_pct)

            if total_events >= MIN_OCCURRENCES:
                pos_pct = len(pos_changes) / total_events * 100
                neg_pct = len(neg_changes) / total_events * 100
                avg_pos_change = np.mean(pos_changes) if pos_changes else 0
                avg_neg_change = np.mean(neg_changes) if neg_changes else 0

                impact_results.append({
                    "Macro Factor": macro_name,
                    "Macro Direction": direction,
                    "Asset": ticker,
                    "Positive Impact %": round(pos_pct, 2),
                    "Negative Impact %": round(neg_pct, 2),
                    "Avg Positive Change %": round(avg_pos_change, 2),
                    "Avg Negative Change %": round(avg_neg_change, 2),
                    "Occurrences": total_events
                })

# --- ESPORTAZIONE FINALE ---
impact_df = pd.DataFrame(impact_results)
print("\n=== RISULTATI COMPLETI FILTRATI (≥ 30 occorrenze) ===")
pd.set_option('display.max_rows', None)
print(impact_df)

impact_df.to_csv("impact_scores_significant.csv", index=False)
