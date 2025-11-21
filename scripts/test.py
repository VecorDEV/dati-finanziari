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
# Ho mappato ogni singolo asset della tua lista originale nel settore corretto.

SECTOR_CONFIG = {
    
    # --- 1. BIG TECH & SOFTWARE (Leader: NASDAQ 100) ---
    "US_BIG_TECH": {
        "benchmark": "^NDX",
        "assets": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "ADBE", "CRM", "ORCL", 
            "IBM", "ACN", "NOW", "INTU", "ADP", "FI", "FIS", "SNOW", "DDOG", 
            "PLTR", "NET", "SQ", "SHOP", "TWLO", "ZM", "ZI", "DUOL", "UBER", 
            "LYFT", "ROKU", "WDAY", "TEAM"
        ]
    },

    # --- 2. SEMICONDUTTORI & HARDWARE (Leader: SOXX ETF) ---
    "SEMICONDUCTORS": {
        "benchmark": "SOXX",
        "assets": [
            "NVDA", "TSLA", "AMD", "INTC", "QCOM", "CSCO", "ADI", "LMT", "TXN", 
            "MU", "AVGO", "ARM", "BTDR", "NAAS" # BTDR e NAAS sono tech/hardware related
        ]
    },

    # --- 3. FINANZA & PAGAMENTI (Leader: FINANCIAL SECTOR ETF) ---
    "FINANCE": {
        "benchmark": "XLF",
        "assets": [
            "JPM", "V", "MA", "PYPL", "MS", "GS", "AXP", "C", "WFC", "BAC", 
            "SCHW", "PNC", "USB", "COIN", "ICE", "MMC", "AON", "TROW", "CME", 
            "SPGI", "MCO", "BRK-B"
        ]
    },

    # --- 4. HEALTHCARE & PHARMA (Leader: HEALTHCARE ETF) ---
    "HEALTHCARE": {
        "benchmark": "XLV",
        "assets": [
            "JNJ", "PFE", "MRK", "ABT", "BMY", "LLY", "AMGN", "CVS", "BDX", 
            "ZTS", "SYK", "EW", "LNTH", "UNH", "ISRG", "TMO", "DHR"
        ]
    },

    # --- 5. CONSUMER & RETAIL (Leader: CONSUMER DISC. ETF) ---
    "CONSUMER_RETAIL": {
        "benchmark": "XLY",
        "assets": [
            "WMT", "KO", "PEP", "MCD", "NKE", "HD", "COST", "SBUX", "LOW", 
            "TGT", "BKNG", "TJX", "CL", "EL", "GM", "F", "TM", "HMC", "TSLA", 
            "RIVN", "LCID", "NIO", "HTZ", "SCHL"
        ]
    },

    # --- 6. INDUSTRIALE & ENERGIA (Leader: INDUSTRIAL ETF) ---
    "INDUSTRIAL_ENERGY": {
        "benchmark": "XLI",
        "assets": [
            "XOM", "CVX", "GE", "CAT", "DE", "HON", "LMT", "ITW", "FDX", "SO", 
            "APD", "D", "PSA", "AEP", "DUK", "NRG", "HE", "PLD", "NSC", "PBR", 
            "VALE" # PBR e VALE sono materie prime/energia
        ]
    },

    # --- 7. MEDIA & TELECOM (Leader: COMM SERVICES ETF) ---
    "MEDIA_TELECOM": {
        "benchmark": "XLC",
        "assets": [
            "DIS", "NFLX", "T", "TMUS", "VZ", "CMCSA", "CHTR", "WBD", "PARA"
        ]
    },

    # --- 8. CINA & EMERGING (Leader: HANG SENG) ---
    "CHINA_ADR": {
        "benchmark": "^HSI",
        "assets": [
            "BABA", "BIDU", "JD", "PDD", "NIO", "TCEHY", "AMX" # AMX è messico ma emerging
        ]
    },

    # --- 9. ITALIA & EUROPA (Leader: FTSE MIB) ---
    "ITALY_EU": {
        "benchmark": "FTSEMIB.MI",
        "assets": [
            "UCG.MI", "ISP.MI", "ENEL.MI", "STLAM.MI", "LDO.MI", "PST.MI", 
            "ENI.MI", "RACE.MI", "TIT.MI", "TRN.MI", "SRG.MI", "MONC.MI"
        ]
    },

    # --- 10. CRYPTO ASSETS (Leader: BITCOIN) ---
    "CRYPTO": {
        "benchmark": "BTC-USD",
        "assets": [
            "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD", "BCH-USD", "EOS-USD", 
            "XLM-USD", "ADA-USD", "TRX-USD", "NEO-USD", "DASH-USD", "XMR-USD", 
            "ETC-USD", "ZEC-USD", "BNB-USD", "DOGE-USD", "USDT-USD", "LINK-USD", 
            "ATOM-USD", "XTZ-USD"
        ]
    },

    # --- 11. FOREX (Leader: DOLLAR INDEX) ---
    "FOREX": {
        "benchmark": "DX-Y.NYB",
        "assets": [
            "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", 
            "USDCHF=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", 
            "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "EURAUD=X", "EURNZD=X", 
            "EURCAD=X", "EURCHF=X", "GBPCHF=X", "AUDCAD=X"
        ]
    },

    # --- 12. COMMODITIES (Leader: GSCI INDEX) ---
    "COMMODITIES": {
        "benchmark": "^SPGSCI",
        "assets": [
            "GC=F", "SI=F", "CL=F", "NG=F", "CC=F", "HG=F", "KC=F", "ZC=F"
        ]
    },

    # --- 13. INDICI GLOBALI (Leader: MSCI WORLD) ---
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
# 2. ENGINE MATEMATICO (Hybrid Scorer)
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
        # Minimo dati necessari
        if len(df) < 30: return 0.0
        
        close = df['Close']
        
        # Adattiamo la media mobile alla lunghezza dello storico disponibile
        window = 200 if len(df) >= 200 else 50
        sma_long = close.rolling(window=window).mean().iloc[-1]
        
        current_price = close.iloc[-1]
        
        # RSI calculation
        if len(df) > 15:
            rsi = self._calculate_rsi(close).iloc[-1]
        else:
            rsi = 50.0

        score = 0.0
        # A. Trend (0.5)
        if current_price > sma_long: score += 0.5
        else: score -= 0.5
        
        # B. Momentum (0.5)
        if rsi < 30: score += 0.5    # Ipervenduto -> Rimbalzo
        elif rsi > 70: score -= 0.5  # Ipercomprato -> Ritracciamento
        
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df_history, news_sentiment, news_count, leader_score, is_leader):
        s_tech = self._get_technical_score(df_history)
        s_news = news_sentiment
        
        # Se l'asset è il leader stesso, il suo "Leader Score" esterno è 0 (si autoregola)
        current_leader_score = 0.0 if is_leader else leader_score
        
        # --- PESI DINAMICI ---
        if is_leader:
            # Modello a 2 fattori (Tecnico + News)
            if news_count == 0: w_n, w_l, w_t = 0.0, 0.0, 1.0
            elif news_count <= 3: w_n, w_l, w_t = 0.30, 0.0, 0.70
            else: w_n, w_l, w_t = 0.60, 0.0, 0.40
        else:
            # Modello a 3 fattori (Tecnico + News + Leader)
            if news_count == 0:
                # Niente news: comanda il trend tecnico e il settore
                w_n, w_l, w_t = 0.0, 0.35, 0.65
            elif news_count <= 3:
                w_n, w_l, w_t = 0.20, 0.25, 0.55
            else:
                # Molte news: il sentiment specifico comanda
                w_n, w_l, w_t = 0.55, 0.15, 0.30
        
        final_score = (s_news * w_n) + (s_tech * w_t) + (current_leader_score * w_l)
        final_score = max(min(final_score, 1.0), -1.0)
        
        # Trasforma in probabilità (50% base + score)
        prob = 50 + (final_score * 50)
        return round(prob, 2), round(s_tech, 2), round(s_news, 2), round(current_leader_score, 2)

# ==============================================================================
# 3. DATA FETCHING (Robustezza Yahoo)
# ==============================================================================

def get_data(ticker):
    """Scarica dati storici. Gestisce il download robusto."""
    try:
        # Scarichiamo 6 mesi di dati per velocità ma sufficienti per trend
        # auto_adjust=True gestisce dividendi e split
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        return df
    except: return pd.DataFrame()

def get_leader_trend(leader_ticker):
    """Calcola il trend del benchmark di settore."""
    try:
        df = get_data(leader_ticker)
        if df.empty or len(df) < 50: return 0.0
        
        close = df['Close']
        # Gestione MultiIndex se presente
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]

        sma_50 = close.rolling(window=50).mean().iloc[-1]
        current = close.iloc[-1]
        
        return 0.5 if current > sma_50 else -0.5
    except: return 0.0

def get_news_data(ticker):
    """Scarica news da Google RSS e analizza il sentiment."""
    # Pulizia nome per migliorare la ricerca
    clean_ticker = ticker.replace("=F", " commodity").replace("=X", " forex").replace("-USD", " crypto")
    
    # URL Google News RSS Search
    rss_url = f"https://news.google.com/rss/search?q={clean_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    
    try:
        resp = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=2)
        root = ET.fromstring(resp.content)
        titles = []
        now = datetime.now().astimezone()
        
        # Filtro temporale 48h
        for item in root.findall('.//item'):
            try:
                pd_date = parsedate_to_datetime(item.find('pubDate').text)
                if (now - pd_date) < timedelta(hours=48):
                    titles.append(item.find('title').text)
            except: continue
            
        count = len(titles)
        if count == 0: return 0.0, 0
        
        # Analisi VADER
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
# 4. ESECUZIONE
# ==============================================================================

if __name__ == "__main__":
    scorer = HybridScorer()
    print(f"\n--- ANALISI COMPLETA PORTAFOGLIO ({datetime.now().strftime('%Y-%m-%d')}) ---")
    
    all_results = []
    leader_cache = {}

    # Iterazione su tutti i settori
    for sector_name, config in SECTOR_CONFIG.items():
        leader_ticker = config['benchmark']
        assets = config['assets']
        
        # 1. Analisi Leader (Cache per non riscaricare se usato più volte)
        if leader_ticker not in leader_cache:
            leader_cache[leader_ticker] = get_leader_trend(leader_ticker)
        
        sector_trend = leader_cache[leader_ticker]
        trend_icon = "📈" if sector_trend > 0 else "📉"
        
        print(f"\n📂 {sector_name} [Leader: {leader_ticker} {trend_icon}]")
        print("-" * 60)

        # 2. Analisi Singoli Asset
        for ticker in assets:
            try:
                df = get_data(ticker)
                
                if not df.empty:
                    sentiment, count = get_news_data(ticker)
                    is_leader = (ticker == leader_ticker)
                    
                    prob, tech_s, news_s, lead_s = scorer.calculate_probability(
                        df, sentiment, count, sector_trend, is_leader
                    )
                    
                    # Segnale testuale
                    if prob >= 60: sig = "STRONG BUY"
                    elif prob >= 53: sig = "BUY"
                    elif prob <= 40: sig = "STRONG SELL"
                    elif prob <= 47: sig = "SELL"
                    else: sig = "HOLD"
                    
                    all_results.append({
                        "Ticker": ticker,
                        "Sector": sector_name,
                        "Score": prob,
                        "Signal": sig,
                        "News": count,
                        "Sent": news_s,
                        "Tech": tech_s,
                        "Trend": lead_s
                    })
                    
                    # Stampa di avanzamento
                    print(f"   {ticker:<10} | {prob}% | {sig:<10} | News: {count}")
                else:
                    print(f"   {ticker:<10} | ⚠️ NO DATA")
            
            except Exception as e:
                print(f"   {ticker:<10} | ❌ ERROR: {e}")
            
            # Pausa minima per evitare Rate Limit di Yahoo (importante con 200 asset)
            time.sleep(0.1)

    # ==========================================================================
    # 5. OUTPUT CLASSIFICA FINALE
    # ==========================================================================
    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res = df_res.sort_values(by="Score", ascending=False)
        
        print("\n\n" + "="*95)
        print(f"🏆 CLASSIFICA GENERALE (Ordinata per Trend Score Decrescente)")
        print("="*95)
        print(f"{'TICKER':<10} | {'SECTOR':<18} | {'SCORE':<6} | {'SIGNAL':<11} | {'NEWS':<4} | {'SENT':<5} | {'TECH':<5} | {'LEAD':<5}")
        print("-" * 95)
        
        for _, row in df_res.iterrows():
            icon = "🟢" if "BUY" in row['Signal'] else "🔴" if "SELL" in row['Signal'] else "⚪"
            print(f"{row['Ticker']:<10} | {row['Sector'][:18]:<18} | {row['Score']:<6} | {icon} {row['Signal']:<9} | {row['News']:<4} | {row['Sent']:<5} | {row['Tech']:<5} | {row['Trend']:<5}")
            
        # Salvataggio su file
        df_res.to_csv("financial_predictions_full.csv", index=False)
        print("\n✅ Analisi completata. Salvato in 'financial_predictions_full.csv'")
    else:
        print("\n❌ Nessun risultato generato. Controlla la connessione o i blocchi IP.")
