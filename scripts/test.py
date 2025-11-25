import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import os
from email.utils import parsedate_to_datetime

# --- SETUP AI ---
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# ==============================================================================
# 1. CONFIGURAZIONE SETTORI COMPLETA (TUTTI I TUOI ASSET)
# ==============================================================================

SECTOR_CONFIG = {
    "US_BIG_TECH": {
        "benchmark": "^NDX",
        "assets": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "ADBE", "CRM", "ORCL", 
            "IBM", "ACN", "NOW", "INTU", "ADP", "FI", "FIS", "SNOW", "DDOG", 
            "PLTR", "NET", "SQ", "SHOP", "TWLO", "ZM", "ZI", "DUOL", "UBER", 
            "LYFT", "ROKU", "WDAY", "TEAM"
        ]
    },
    "SEMICONDUCTORS": {
        "benchmark": "SOXX",
        "assets": [
            "NVDA", "TSLA", "AMD", "INTC", "QCOM", "CSCO", "ADI", "LMT", "TXN", 
            "MU", "AVGO", "ARM", "BTDR", "NAAS"
        ]
    },
    "FINANCE": {
        "benchmark": "XLF",
        "assets": [
            "JPM", "V", "MA", "PYPL", "MS", "GS", "AXP", "C", "WFC", "BAC", 
            "SCHW", "PNC", "USB", "COIN", "ICE", "MMC", "AON", "TROW", "CME", 
            "SPGI", "MCO", "BRK-B"
        ]
    },
    "HEALTHCARE": {
        "benchmark": "XLV",
        "assets": [
            "JNJ", "PFE", "MRK", "ABT", "BMY", "LLY", "AMGN", "CVS", "BDX", 
            "ZTS", "SYK", "EW", "LNTH", "UNH", "ISRG", "TMO", "DHR"
        ]
    },
    "CONSUMER_RETAIL": {
        "benchmark": "XLY",
        "assets": [
            "WMT", "KO", "PEP", "MCD", "NKE", "HD", "COST", "SBUX", "LOW", 
            "TGT", "BKNG", "TJX", "CL", "EL", "GM", "F", "TM", "HMC", "TSLA", 
            "RIVN", "LCID", "NIO", "HTZ", "SCHL"
        ]
    },
    "INDUSTRIAL_ENERGY": {
        "benchmark": "XLI",
        "assets": [
            "XOM", "CVX", "GE", "CAT", "DE", "HON", "LMT", "ITW", "FDX", "SO", 
            "APD", "D", "PSA", "AEP", "DUK", "NRG", "HE", "PLD", "NSC", "PBR", 
            "VALE"
        ]
    },
    "MEDIA_TELECOM": {
        "benchmark": "XLC",
        "assets": [
            "DIS", "NFLX", "T", "TMUS", "VZ", "CMCSA", "CHTR", "WBD", "PARA"
        ]
    },
    "CHINA_ADR": {
        "benchmark": "^HSI",
        "assets": [
            "BABA", "BIDU", "JD", "PDD", "NIO", "TCEHY", "AMX"
        ]
    },
    "ITALY_EU": {
        "benchmark": "FTSEMIB.MI",
        "assets": [
            "UCG.MI", "ISP.MI", "ENEL.MI", "STLAM.MI", "LDO.MI", "PST.MI", 
            "ENI.MI", "RACE.MI", "TIT.MI", "TRN.MI", "SRG.MI", "MONC.MI"
        ]
    },
    "CRYPTO": {
        "benchmark": "BTC-USD",
        "assets": [
            "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD", "BCH-USD", "EOS-USD", 
            "XLM-USD", "ADA-USD", "TRX-USD", "NEO-USD", "DASH-USD", "XMR-USD", 
            "ETC-USD", "ZEC-USD", "BNB-USD", "DOGE-USD", "USDT-USD", "LINK-USD", 
            "ATOM-USD", "XTZ-USD"
        ]
    },
    "FOREX": {
        "benchmark": "DX-Y.NYB",
        "assets": [
            "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", 
            "USDCHF=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", 
            "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "EURAUD=X", "EURNZD=X", 
            "EURCAD=X", "EURCHF=X", "GBPCHF=X", "AUDCAD=X"
        ]
    },
    "COMMODITIES": {
        "benchmark": "^SPGSCI",
        "assets": [
            "GC=F", "SI=F", "CL=F", "NG=F", "CC=F", "HG=F", "KC=F", "ZC=F"
        ]
    },
    "INDICES": {
        "benchmark": "URTH",
        "assets": [
            "^GSPC", "^DJI", "^NDX", "^IXIC", "^RUT", "^VIX", "^STOXX50E", 
            "FTSEMIB.MI", "^GDAXI", "^FTSE", "^FCHI", "^N225", "^HSI", 
            "^IBEX", "^AEX", "^SSMI"
        ]
    }
}

# ==============================================================================
# 2. DATA FETCHING ROBUSTO (IL FIX)
# ==============================================================================

def get_data(ticker):
    """
    Scarica dati e corregge il problema 'MultiIndex' di Yahoo Finance.
    Questa è la parte cruciale per evitare il crash.
    """
    try:
        # Scarichiamo 6 mesi di dati
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        
        if df.empty:
            return pd.DataFrame()

        # FIX: Se ci sono MultiIndex nelle colonne (es. ('Close', 'AAPL')), appiattiscili
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # Se c'è un livello ticker, droppalo per avere solo 'Close'
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)
            except:
                pass
        
        # Verifica finale che 'Close' esista
        if 'Close' not in df.columns:
            return pd.DataFrame()
            
        return df
    except Exception:
        # Silenzia l'errore per continuare con gli altri asset
        return pd.DataFrame()

def get_leader_trend(leader_ticker):
    """Calcola trend leader (Benchmark)."""
    try:
        df = get_data(leader_ticker)
        if df.empty or len(df) < 50: return 0.0
        
        close = df['Close']
        
        # Calcolo sicuro con float nativi
        sma_50 = float(close.rolling(window=50).mean().iloc[-1])
        current = float(close.iloc[-1])
        
        return 0.5 if current > sma_50 else -0.5
    except: return 0.0

# ==============================================================================
# 3. ENGINE MATEMATICO
# ==============================================================================

class HybridScorer:
    def _calculate_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        loss = loss.replace(0, np.nan) 
        rs = gain / loss
        rs = rs.fillna(0)
        return 100 - (100 / (1 + rs))

    def _get_technical_score(self, df):
        if len(df) < 30: return 0.0
        close = df['Close']
        
        window = 200 if len(df) >= 200 else 50
        
        try:
            sma_long = float(close.rolling(window=window).mean().iloc[-1])
            current_price = float(close.iloc[-1])
            
            if len(df) > 15:
                rsi = float(self._calculate_rsi(close).iloc[-1])
            else:
                rsi = 50.0
        except:
            return 0.0

        score = 0.0
        
        # Logica Trend
        if current_price > sma_long: score += 0.5
        else: score -= 0.5
        
        # Logica Momentum
        if rsi < 30: score += 0.5
        elif rsi > 70: score -= 0.5
        
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df_history, news_sentiment, news_count, leader_score, is_leader):
        s_tech = self._get_technical_score(df_history)
        s_news = news_sentiment
        
        current_leader = 0.0 if is_leader else leader_score
        
        # Pesi Dinamici (Confidence Score)
        if is_leader:
            if news_count == 0: w_n, w_l, w_t = 0.0, 0.0, 1.0
            elif news_count <= 3: w_n, w_l, w_t = 0.30, 0.0, 0.70
            else: w_n, w_l, w_t = 0.60, 0.0, 0.40
        else:
            if news_count == 0: w_n, w_l, w_t = 0.0, 0.35, 0.65
            elif news_count <= 3: w_n, w_l, w_t = 0.20, 0.25, 0.55
            else: w_n, w_l, w_t = 0.55, 0.15, 0.30
        
        final_score = (s_news * w_n) + (s_tech * w_t) + (current_leader * w_l)
        final_score = max(min(final_score, 1.0), -1.0)
        
        return round(50 + (final_score * 50), 2), round(s_tech, 2), round(s_news, 2), round(current_leader, 2)

# ==============================================================================
# 4. NEWS FETCHING
# ==============================================================================

def get_news_data(ticker):
    clean = ticker.replace("=F", " commodity").replace("=X", " forex").replace("-USD", " crypto")
    rss_url = f"https://news.google.com/rss/search?q={clean}+stock&hl=en-US&gl=US&ceid=US:en"
    
    try:
        resp = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=2)
        root = ET.fromstring(resp.content)
        titles = []
        now = datetime.now().astimezone()
        
        for item in root.findall('.//item'):
            try:
                pd_date = parsedate_to_datetime(item.find('pubDate').text)
                if (now - pd_date) < timedelta(hours=48):
                    titles.append(item.find('title').text)
            except: continue
            
        count = len(titles)
        if count == 0: return 0.0, 0
        
        sia = SentimentIntensityAnalyzer()
        lexicon = {
            'surge': 4.0, 'jump': 2.0, 'rally': 3.5, 'soar': 4.0, 'bull': 3.0, 'beat': 2.5,
            'plunge': -4.0, 'crash': -4.0, 'drop': -3.0, 'slump': -3.5, 'bear': -3.0,
            'inflation': -1.5, 'recession': -3.0
        }
        sia.lexicon.update(lexicon)
        total = sum([sia.polarity_scores(t)['compound'] for t in titles])
        return (total / count), count
    except: return 0.0, 0

# ==============================================================================
# 5. ESECUZIONE
# ==============================================================================

if __name__ == "__main__":
    scorer = HybridScorer()
    print(f"\n--- ANALISI PORTAFOGLIO SETTORIALE ({datetime.now().strftime('%Y-%m-%d')}) ---")
    
    all_results = []
    leader_cache = {}

    for sector, data in SECTOR_CONFIG.items():
        leader = data['benchmark']
        assets = data['assets']
        
        # 1. Analisi Leader (Cache)
        if leader not in leader_cache:
            leader_cache[leader] = get_leader_trend(leader)
        
        sector_trend = leader_cache[leader]
        trend_icon = "📈" if sector_trend > 0 else "📉"
        
        print(f"\n📂 {sector} [Leader: {leader} {trend_icon}]")
        print("-" * 60)

        # 2. Analisi Asset
        for ticker in assets:
            df = get_data(ticker)
            
            if not df.empty:
                sentiment, count = get_news_data(ticker)
                is_leader = (ticker == leader)
                
                prob, tech, sent, lead = scorer.calculate_probability(
                    df, sentiment, count, sector_trend, is_leader
                )
                
                if prob >= 60: sig = "STRONG BUY"
                elif prob >= 53: sig = "BUY"
                elif prob <= 40: sig = "STRONG SELL"
                elif prob <= 47: sig = "SELL"
                else: sig = "HOLD"
                
                all_results.append({
                    "Ticker": ticker,
                    "Sector": sector,
                    "Score": prob,
                    "Signal": sig,
                    "News": count,
                    "Sent": sent,
                    "Tech": tech,
                    "Trend": lead
                })
                print(f"   {ticker:<10} | {prob}% | {sig:<11} | News:{count}")
            else:
                print(f"   {ticker:<10} | ⚠️ NO DATA ({ticker})")
            
            time.sleep(0.05)

    # OUTPUT FINALE
    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res = df_res.sort_values(by="Score", ascending=False)
        
        print("\n\n" + "="*100)
        print(f"🏆 CLASSIFICA FINALE (Ordinata per Score)")
        print("="*100)
        print(f"{'TICKER':<10} | {'SECTOR':<15} | {'SCORE':<6} | {'SIGNAL':<11} | {'NEWS':<4} | {'SENT':<5} | {'TECH':<5} | {'LEAD':<5}")
        print("-" * 100)
        
        for _, row in df_res.iterrows():
            icon = "🟢" if "BUY" in row['Signal'] else "🔴" if "SELL" in row['Signal'] else "⚪"
            print(f"{row['Ticker']:<10} | {row['Sector'][:15]:<15} | {row['Score']:<6} | {icon} {row['Signal']:<9} | {row['News']:<4} | {row['Sent']:<5} | {row['Tech']:<5} | {row['Trend']:<5}")
            
        df_res.to_csv("predictions_sectors.csv", index=False)
        print("\n✅ Analisi completata. Salvato in 'predictions_sectors.csv'")
    else:
        print("\n❌ Nessun risultato generato.")
