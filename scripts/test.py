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
# 1. LA TUA LISTA COMPLETA (RAW)
# ==============================================================================
FULL_ASSET_LIST = [
    # Stocks USA & Global
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "V", "JPM", "JNJ", "WMT",
    "NVDA", "PYPL", "DIS", "NFLX", "NIO", "NRG", "ADBE", "INTC", "CSCO", "PFE",
    "KO", "PEP", "MRK", "ABT", "XOM", "CVX", "T", "MCD", "NKE", "HD",
    "IBM", "CRM", "BMY", "ORCL", "ACN", "LLY", "QCOM", "HON", "COST", "SBUX",
    "CAT", "LOW", "MS", "GS", "AXP", "INTU", "AMGN", "GE", "FIS", "CVS",
    "DE", "BDX", "NOW", "SCHW", "LMT", "ADP", "C", "PLD", "NSC", "TMUS",
    "ITW", "FDX", "PNC", "SO", "APD", "ADI", "ICE", "ZTS", "TJX", "CL",
    "MMC", "EL", "GM", "CME", "EW", "AON", "D", "PSA", "AEP", "TROW", 
    "LNTH", "HE", "BTDR", "NAAS", "SCHL", "TGT", "SYK", "BKNG", "DUK", "USB",
    "ARM", "BABA", "BIDU", "COIN", "DDOG", "HTZ", "JD", "LCID", "LYFT", "NET", 
    "PDD", "PLTR", "RIVN", "ROKU", "SHOP", "SNOW", "SQ", "TWLO", "UBER", "ZI", 
    "ZM", "DUOL", "PBR", "VALE", "AMX",
    
    # Italia / Europa
    "ISP.MI", "ENEL.MI", "STLAM.MI", "LDO.MI", "PST.MI", "UCG.MI",

    # Forex
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", 
    "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", 
    "CHFJPY=X", "EURAUD=X", "EURNZD=X", "EURCAD=X", "EURCHF=X", "GBPCHF=X", 
    "AUDCAD=X",

    # Indices (Global)
    "^GSPC", "^DJI", "^NDX", "^IXIC", "^RUT", "^VIX", "^STOXX50E", "FTSEMIB.MI", 
    "^GDAXI", "^FTSE", "^FCHI", "^N225", "^HSI",

    # Crypto
    "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD", "BCH-USD", "EOS-USD", "XLM-USD", 
    "ADA-USD", "TRX-USD", "NEO-USD", "DASH-USD", "XMR-USD", "ETC-USD", "ZEC-USD", 
    "BNB-USD", "DOGE-USD", "USDT-USD", "LINK-USD", "ATOM-USD", "XTZ-USD",

    # Commodities
    "CC=F", "GC=F", "SI=F", "CL=F", "NG=F"
]

# ==============================================================================
# 2. AUTO-CATEGORIZZATORE (Distribuisce gli asset nei settori)
# ==============================================================================
def categorize_assets(asset_list):
    sectors = {
        "CRYPTO": {"benchmark": "BTC-USD", "assets": []},
        "FOREX": {"benchmark": "DX-Y.NYB", "assets": []},
        "COMMODITIES": {"benchmark": "^SPGSCI", "assets": []},
        "ITALY_EU": {"benchmark": "FTSEMIB.MI", "assets": []},
        "INDICES": {"benchmark": "URTH", "assets": []}, # URTH è ETF World
        "US_STOCKS": {"benchmark": "^GSPC", "assets": []} # Default fallback
    }
    
    # Lista speciale Tech per assegnare il Nasdaq invece dell'S&P500
    tech_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", 
                 "AMD", "INTC", "QCOM", "CRM", "ADBE", "PLTR", "COIN", "SHOP", "SQ"]

    for ticker in asset_list:
        # 1. Crypto
        if "-USD" in ticker:
            sectors["CRYPTO"]["assets"].append(ticker)
        # 2. Forex
        elif "=X" in ticker:
            sectors["FOREX"]["assets"].append(ticker)
        # 3. Commodities
        elif "=F" in ticker:
            sectors["COMMODITIES"]["assets"].append(ticker)
        # 4. Italia
        elif ".MI" in ticker:
            # FTSEMIB.MI è sia un asset che un benchmark. 
            # Lo mettiamo negli INDICES se inizia con ^ o è un indice noto, 
            # altrimenti in ITALY_EU
            if ticker == "FTSEMIB.MI": 
                sectors["INDICES"]["assets"].append(ticker)
            else:
                sectors["ITALY_EU"]["assets"].append(ticker)
        # 5. Indici
        elif ticker.startswith("^"):
            sectors["INDICES"]["assets"].append(ticker)
        # 6. Stocks (Default)
        else:
            # Raffinamento Tech vs General
            if ticker in tech_list:
                # Creiamo il settore Tech al volo se serve o usiamo US_STOCKS
                # Per semplicità qui usiamo US_STOCKS ma cambiamo il benchmark a runtime se vuoi
                sectors["US_STOCKS"]["assets"].append(ticker)
            else:
                sectors["US_STOCKS"]["assets"].append(ticker)
    
    return sectors

# ==============================================================================
# 3. ENGINE DI PREVISIONE
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
        if len(df) < 200: return 0.0
        close = df['Close']
        sma_200 = close.rolling(window=200).mean().iloc[-1]
        rsi = self._calculate_rsi(close).iloc[-1]
        current_price = close.iloc[-1]

        score = 0.0
        if current_price > sma_200: score += 0.5
        else: score -= 0.5
        if rsi < 30: score += 0.5
        elif rsi > 70: score -= 0.5
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df_history, news_sentiment, news_count, leader_score, is_leader):
        s_tech = self._get_technical_score(df_history)
        s_news = news_sentiment
        
        # --- GESTIONE LOGICA "CANE CHE SI MORDE LA CODA" ---
        if is_leader:
            s_leader = 0.0 # Se sono io il leader, il mio "leader score" esterno è 0
            # Aggiustiamo i pesi perché manca un fattore
            if news_count == 0:
                w_n, w_l, w_t = 0.0, 0.0, 1.0 # Solo Tecnico
            elif news_count <= 3:
                w_n, w_l, w_t = 0.30, 0.0, 0.70
            else:
                w_n, w_l, w_t = 0.60, 0.0, 0.40
        else:
            s_leader = leader_score
            # Pesi Standard a 3 Fattori
            if news_count == 0:
                w_n, w_l, w_t = 0.0, 0.35, 0.65
            elif news_count <= 3:
                w_n, w_l, w_t = 0.25, 0.25, 0.50
            else:
                w_n, w_l, w_t = 0.55, 0.15, 0.30
        
        final_score = (s_news * w_n) + (s_tech * w_t) + (s_leader * w_l)
        final_score = max(min(final_score, 1.0), -1.0)
        
        return {
            'prob': round(50 + (final_score * 50), 2),
            'tech': round(s_tech, 2),
            'news': round(s_news, 2),
            'leader': round(s_leader, 2)
        }

# ==============================================================================
# 4. FUNZIONI DATA & NEWS
# ==============================================================================
def get_data(ticker):
    try:
        return yf.Ticker(ticker).history(period="1y", progress=False)
    except: return pd.DataFrame()

def get_leader_trend(leader_ticker):
    try:
        df = get_data(leader_ticker)
        if df.empty: return 0.0
        close = df['Close']
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        current = close.iloc[-1]
        return 0.5 if current > sma_50 else -0.5
    except: return 0.0

def get_news_data(ticker):
    # Mapping per aiutare la ricerca news
    clean_ticker = ticker.replace("=F", " futures").replace("=X", " forex").replace("-USD", " crypto")
    rss_url = f"https://news.google.com/rss/search?q={clean_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    
    try:
        resp = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
        root = ET.fromstring(resp.content)
        titles = []
        now = datetime.now().astimezone()
        
        for item in root.findall('.//item'):
            try:
                pd = parsedate_to_datetime(item.find('pubDate').text)
                if (now - pd) < timedelta(hours=48):
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
    
    # 1. Organizza gli asset automaticamente
    SECTOR_CONFIG = categorize_assets(FULL_ASSET_LIST)
    
    print(f"\n--- ANALISI COMPLETA SU {len(FULL_ASSET_LIST)} ASSET ---")
    
    results = []
    leader_cache = {}

    for sector, data in SECTOR_CONFIG.items():
        leader = data['benchmark']
        assets = data['assets']
        
        # Calcolo Leader
        if leader not in leader_cache:
            leader_cache[leader] = get_leader_trend(leader)
        leader_score = leader_cache[leader]
        
        print(f"\n📂 SETTORE: {sector} (Leader: {leader} | Trend: {leader_score})")

        for ticker in assets:
            df = get_data(ticker)
            
            if not df.empty:
                sentiment, count = get_news_data(ticker)
                
                # Controlla se l'asset è il leader stesso per evitare conteggio doppio
                is_leader = (ticker == leader)
                
                res = scorer.calculate_probability(df, sentiment, count, leader_score, is_leader)
                
                sig = "BUY 🟢" if res['prob'] > 55 else "SELL 🔴" if res['prob'] < 45 else "HOLD ⚪"
                
                results.append({
                    "Ticker": ticker,
                    "Score": res['prob'],
                    "Signal": sig,
                    "News": count,
                    "Sent": res['news'],
                    "Tech": res['tech']
                })
                print(".", end="", flush=True)
            else:
                print("x", end="", flush=True)
            time.sleep(0.1)

    # OUTPUT FINALE
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="Score", ascending=False)
    
    print("\n\n" + "="*85)
    print(f"🏆 CLASSIFICA ASSET (Top Opportunities)")
    print("="*85)
    print(f"{'TICKER':<12} | {'SCORE':<6} | {'SIGNAL':<8} | {'NEWS':<4} | {'SENT':<6} | {'TECH':<6}")
    print("-" * 85)
    
    for _, row in df_res.iterrows():
        print(f"{row['Ticker']:<12} | {row['Score']:<6} | {row['Signal']:<8} | {row['News']:<4} | {row['Sent']:<6} | {row['Tech']:<6}")
