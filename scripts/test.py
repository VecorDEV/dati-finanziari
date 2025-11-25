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
# 1. DATI UTENTE & KEYWORDS
# ==============================================================================

USER_SYMBOL_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "V", "JPM", "JNJ", "WMT",
    "NVDA", "PYPL", "DIS", "NFLX", "NIO", "NRG", "ADBE", "INTC", "CSCO", "PFE",
    "KO", "PEP", "MRK", "ABT", "XOM", "CVX", "T", "MCD", "NKE", "HD",
    "IBM", "CRM", "BMY", "ORCL", "ACN", "LLY", "QCOM", "HON", "COST", "SBUX",
    "CAT", "LOW", "MS", "GS", "AXP", "INTU", "AMGN", "GE", "FIS", "CVS",
    "DE", "BDX", "NOW", "SCHW", "LMT", "ADP", "C", "PLD", "NSC", "TMUS",
    "ITW", "FDX", "PNC", "SO", "APD", "ADI", "ICE", "ZTS", "TJX", "CL",
    "MMC", "EL", "GM", "CME", "EW", "AON", "D", "PSA", "AEP", "TROW", 
    "LNTH", "HE", "BTDR", "NAAS", "SCHL", "TGT", "SYK", "BKNG", "DUK", "USB", 
    "ISP.MI", "ENEL.MI", "STLAM.MI", "LDO.MI", 
    "ARM", "BABA", "BIDU", "COIN", "PST.MI", "UCG.MI", "DDOG", "HTZ", "JD", "LCID", "LYFT", "NET", "PDD", 
    "PLTR", "RIVN", "ROKU", "SHOP", "SNOW", "SQ", "TWLO", "UBER", "ZI", "ZM", "DUOL",
    "PBR", "VALE", "AMX",
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
    "AUDJPY", "CADJPY", "CHFJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "GBPCHF", "AUDCAD",
    "SPX500", "DJ30", "NAS100", "NASCOMP", "RUS2000", "VIX", "EU50", "ITA40", "GER40", "UK100",
    "FRA40", "SWI20", "ESP35", "NETH25", "JPN225", "HKG50", "CHN50", "IND50", "KOR200",
    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BCHUSD", "EOSUSD", "XLMUSD", "ADAUSD", "TRXUSD", "NEOUSD",
    "DASHUSD", "XMRUSD", "ETCUSD", "ZECUSD", "BNBUSD", "DOGEUSD", "USDTUSD", "LINKUSD", "ATOMUSD", "XTZUSD",
    "COCOA", "GOLD", "SILVER", "OIL", "NATGAS"
]

# Parole chiave aggiuntive per trovare più news
SYMBOL_KEYWORDS = {
    "OIL": ["Crude Oil", "WTI", "Brent", "Petroleum"],
    "GOLD": ["Gold Price", "XAUUSD", "Precious Metals"],
    "BTCUSD": ["Bitcoin", "Crypto", "Satoshi"],
    "ETHUSD": ["Ethereum", "Vitalik", "DeFi"],
    "TSLA": ["Tesla", "Elon Musk", "EV"],
    "NVDA": ["Nvidia", "AI chips", "Jensen Huang"],
    "COCOA": ["Cocoa futures", "Chocolate price"],
    "NATGAS": ["Natural Gas", "Henry Hub", "LNG"],
    "EURUSD": ["EUR/USD", "Euro Dollar", "ECB"],
    "AAPL": ["Apple", "iPhone", "MacBook"],
    "AMZN": ["Amazon", "AWS", "Prime"],
    "ISP.MI": ["Intesa Sanpaolo", "Carlo Messina"],
    "UCG.MI": ["Unicredit", "Andrea Orcel"]
}

# ==============================================================================
# 2. MAPPATURA GRANULARE DEI SETTORI (CONFIGURAZIONE)
# ==============================================================================

# Definizione dei Benchmark per ogni settore
SECTOR_BENCHMARKS = {
    "BIG_TECH": "^NDX",          # Nasdaq 100
    "SEMICONDUCTORS": "SOXX",    # ETF Semiconduttori
    "ECOMMERCE": "ONLN",         # ETF Online Retail
    "ENTERTAINMENT": "XLC",      # ETF Communication (Disney, Netflix)
    "EV_AUTO": "DRIV",           # ETF Auto Elettriche
    "SOFTWARE_CLOUD": "IGV",     # ETF Tech-Software
    "FINANCE": "XLF",            # ETF Finanziari
    "HEALTHCARE": "XLV",         # ETF Salute
    "CONSUMER": "XLY",           # ETF Consumi Discrezionali
    "ENERGY_IND": "XLI",         # ETF Industriali/Energia
    "CHINA": "^HSI",             # Hang Seng
    "ITALY": "FTSEMIB.MI",       # FTSE MIB
    "CRYPTO": "BTC-USD",         # Bitcoin
    "FOREX": "DX-Y.NYB",         # Dollar Index
    "COMMODITIES": "^SPGSCI",    # Indice Materie Prime
    "INDICES": "URTH"            # MSCI World
}

# Mappatura manuale dei tuoi asset nei settori specifici
ASSET_SECTOR_MAP = {
    # BIG TECH
    "AAPL": "BIG_TECH", "MSFT": "BIG_TECH", "GOOGL": "BIG_TECH", "AMZN": "BIG_TECH", 
    "META": "BIG_TECH", "IBM": "BIG_TECH", "ACN": "BIG_TECH", "CSCO": "BIG_TECH",
    
    # SEMICONDUCTORS
    "NVDA": "SEMICONDUCTORS", "INTC": "SEMICONDUCTORS", "QCOM": "SEMICONDUCTORS", 
    "ADI": "SEMICONDUCTORS", "ARM": "SEMICONDUCTORS", "BTDR": "SEMICONDUCTORS", 
    "NAAS": "SEMICONDUCTORS",
    
    # ECOMMERCE / INTERNET
    "SHOP": "ECOMMERCE", "EBAY": "ECOMMERCE", "JD": "ECOMMERCE", "PDD": "ECOMMERCE", 
    "BABA": "ECOMMERCE", "BIDU": "ECOMMERCE", "UBER": "ECOMMERCE", "LYFT": "ECOMMERCE", 
    "BKNG": "ECOMMERCE",
    
    # ENTERTAINMENT / MEDIA
    "DIS": "ENTERTAINMENT", "NFLX": "ENTERTAINMENT", "ROKU": "ENTERTAINMENT", 
    "T": "ENTERTAINMENT", "TMUS": "ENTERTAINMENT", "DUOL": "ENTERTAINMENT",
    
    # SOFTWARE / CLOUD
    "ADBE": "SOFTWARE_CLOUD", "CRM": "SOFTWARE_CLOUD", "ORCL": "SOFTWARE_CLOUD", 
    "NOW": "SOFTWARE_CLOUD", "INTU": "SOFTWARE_CLOUD", "SNOW": "SOFTWARE_CLOUD", 
    "DDOG": "SOFTWARE_CLOUD", "PLTR": "SOFTWARE_CLOUD", "NET": "SOFTWARE_CLOUD", 
    "SQ": "SOFTWARE_CLOUD", "TWLO": "SOFTWARE_CLOUD", "ZM": "SOFTWARE_CLOUD", 
    "ZI": "SOFTWARE_CLOUD", "ADP": "SOFTWARE_CLOUD", "FIS": "SOFTWARE_CLOUD",
    
    # EV / AUTO
    "TSLA": "EV_AUTO", "NIO": "EV_AUTO", "RIVN": "EV_AUTO", "LCID": "EV_AUTO", 
    "GM": "EV_AUTO", "F": "EV_AUTO", "HTZ": "EV_AUTO", "STLAM.MI": "EV_AUTO",
    
    # FINANCE
    "JPM": "FINANCE", "V": "FINANCE", "MA": "FINANCE", "PYPL": "FINANCE", 
    "MS": "FINANCE", "GS": "FINANCE", "AXP": "FINANCE", "C": "FINANCE", 
    "SCHW": "FINANCE", "PNC": "FINANCE", "USB": "FINANCE", "COIN": "FINANCE", 
    "ICE": "FINANCE", "MMC": "FINANCE", "AON": "FINANCE", "TROW": "FINANCE", 
    "CME": "FINANCE",
    
    # HEALTHCARE
    "JNJ": "HEALTHCARE", "PFE": "HEALTHCARE", "MRK": "HEALTHCARE", "ABT": "HEALTHCARE", 
    "BMY": "HEALTHCARE", "LLY": "HEALTHCARE", "AMGN": "HEALTHCARE", "CVS": "HEALTHCARE", 
    "BDX": "HEALTHCARE", "ZTS": "HEALTHCARE", "SYK": "HEALTHCARE", "EW": "HEALTHCARE", 
    "LNTH": "HEALTHCARE",
    
    # CONSUMER
    "WMT": "CONSUMER", "KO": "CONSUMER", "PEP": "CONSUMER", "MCD": "CONSUMER", 
    "NKE": "CONSUMER", "HD": "CONSUMER", "COST": "CONSUMER", "SBUX": "CONSUMER", 
    "LOW": "CONSUMER", "TGT": "CONSUMER", "TJX": "CONSUMER", "CL": "CONSUMER", 
    "EL": "CONSUMER", "SCHL": "CONSUMER",
    
    # INDUSTRIAL / ENERGY
    "XOM": "ENERGY_IND", "CVX": "ENERGY_IND", "GE": "ENERGY_IND", "CAT": "ENERGY_IND", 
    "DE": "ENERGY_IND", "HON": "ENERGY_IND", "LMT": "ENERGY_IND", "ITW": "ENERGY_IND", 
    "FDX": "ENERGY_IND", "SO": "ENERGY_IND", "APD": "ENERGY_IND", "D": "ENERGY_IND", 
    "PSA": "ENERGY_IND", "AEP": "ENERGY_IND", "DUK": "ENERGY_IND", "NRG": "ENERGY_IND", 
    "HE": "ENERGY_IND", "PLD": "ENERGY_IND", "NSC": "ENERGY_IND", "PBR": "ENERGY_IND", 
    "VALE": "ENERGY_IND",
    
    # ITALY
    "ISP.MI": "ITALY", "ENEL.MI": "ITALY", "LDO.MI": "ITALY", 
    "PST.MI": "ITALY", "UCG.MI": "ITALY",
    
    # OTHERS
    "AMX": "CHINA", # Emerging proxy
}

# ==============================================================================
# 3. MAPPER E FUNZIONI UTILITY
# ==============================================================================

def get_sector_info(yahoo_ticker, original_symbol):
    """Determina il settore e il benchmark per un asset."""
    
    # 1. Controllo mappatura manuale specifica (Stocks)
    clean_orig = original_symbol.upper()
    if clean_orig in ASSET_SECTOR_MAP:
        sector = ASSET_SECTOR_MAP[clean_orig]
        return sector, SECTOR_BENCHMARKS[sector]
    
    # 2. Regole automatiche per classi di asset
    if "-USD" in yahoo_ticker and "BTC-USD" not in yahoo_ticker:
        return "CRYPTO", SECTOR_BENCHMARKS["CRYPTO"]
    
    if "=X" in yahoo_ticker:
        return "FOREX", SECTOR_BENCHMARKS["FOREX"]
        
    if "=F" in yahoo_ticker:
        return "COMMODITIES", SECTOR_BENCHMARKS["COMMODITIES"]
        
    if yahoo_ticker.startswith("^") or "FTSEMIB" in yahoo_ticker:
        return "INDICES", SECTOR_BENCHMARKS["INDICES"]
        
    # 3. Fallback Default
    if "BTC-USD" in yahoo_ticker: return "CRYPTO", "BTC-USD" # Leader
    
    return "US_GENERAL", "^GSPC"

def map_symbol_to_yahoo(symbol):
    s = symbol.upper()
    # Indici
    idx = {
        "SPX500":"^GSPC", "DJ30":"^DJI", "NAS100":"^NDX", "NASCOMP":"^IXIC",
        "RUS2000":"^RUT", "VIX":"^VIX", "EU50":"^STOXX50E", "ITA40":"FTSEMIB.MI",
        "GER40":"^GDAXI", "UK100":"^FTSE", "FRA40":"^FCHI", "SWI20":"^SSMI",
        "ESP35":"^IBEX", "NETH25":"^AEX", "JPN225":"^N225", "HKG50":"^HSI",
        "CHN50":"000001.SS", "IND50":"^NSEI", "KOR200":"^KS11"
    }
    if s in idx: return idx[s]

    # Commodities
    comm = {"GOLD":"GC=F", "SILVER":"SI=F", "OIL":"CL=F", "NATGAS":"NG=F", "COCOA":"CC=F"}
    if s in comm: return comm[s]

    # Forex
    if len(s)==6 and s.isalpha() and ("USD" in s or "EUR" in s or "JPY" in s): return f"{s}=X"

    # Crypto
    if s.endswith("USD") and len(s)>3 and "USDCAD" not in s: return s.replace("USD", "-USD")

    return s # Default (Stocks)

def get_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        # Fix MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if df.columns.nlevels > 1: df.columns = df.columns.droplevel(1)
            except: pass
        if 'Close' not in df.columns: return pd.DataFrame()
        return df
    except: return pd.DataFrame()

def get_leader_trend(leader_ticker):
    try:
        df = get_data(leader_ticker)
        if df.empty or len(df) < 30: return 0.0
        close = df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        sma = float(close.rolling(window=50).mean().iloc[-1])
        curr = float(close.iloc[-1])
        return 0.5 if curr > sma else -0.5
    except: return 0.0

def get_news_data_advanced(ticker, original_symbol, keyword_map):
    """Ricerca news con parole chiave multiple."""
    search_terms = [ticker.replace("=F","").replace("=X","").replace("-USD","")]
    if original_symbol in keyword_map:
        search_terms.extend(keyword_map[original_symbol])
    
    # Query limitata a 3 termini
    final_query = " OR ".join(search_terms[:3])
    rss_url = f"https://news.google.com/rss/search?q=({final_query})+stock&hl=en-US&gl=US&ceid=US:en"
    
    try:
        resp = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=2)
        root = ET.fromstring(resp.content)
        titles = []
        now = datetime.now().astimezone()
        for item in root.findall('.//item'):
            try:
                pd = parsedate_to_datetime(item.find('pubDate').text)
                if (now - pd) < timedelta(hours=48): titles.append(item.find('title').text)
            except: continue
        
        count = len(titles)
        if count == 0: return 0.0, 0
        
        sia = SentimentIntensityAnalyzer()
        lexicon = {'surge': 4.0, 'jump': 2.0, 'rally': 3.5, 'soar': 4.0, 'bull': 3.0, 'plunge': -4.0, 'crash': -4.0, 'drop': -3.0}
        sia.lexicon.update(lexicon)
        total = sum([sia.polarity_scores(t)['compound'] for t in titles])
        return (total / count), count
    except: return 0.0, 0

# ==============================================================================
# 4. ENGINE
# ==============================================================================

class HybridScorer:
    def _calculate_rsi(self, series, period=14):
        if isinstance(series, pd.DataFrame): series = series.iloc[:, 0]
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
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        
        win = 200 if len(df) >= 200 else 50
        try:
            sma = float(close.rolling(window=win).mean().iloc[-1])
            curr = float(close.iloc[-1])
            rsi = float(self._calculate_rsi(close).iloc[-1]) if len(df)>15 else 50.0
        except: return 0.0

        score = 0.0
        if curr > sma: score += 0.5
        else: score -= 0.5
        if rsi < 30: score += 0.5
        elif rsi > 70: score -= 0.5
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df, sent, news_n, lead, is_lead):
        tech = self._get_technical_score(df)
        curr_lead = 0.0 if is_lead else lead
        
        if is_lead:
            if news_n == 0: w_n, w_l, w_t = 0.0, 0.0, 1.0
            elif news_n <= 3: w_n, w_l, w_t = 0.30, 0.0, 0.70
            else: w_n, w_l, w_t = 0.60, 0.0, 0.40
        else:
            if news_n == 0: w_n, w_l, w_t = 0.0, 0.35, 0.65
            elif news_n <= 3: w_n, w_l, w_t = 0.20, 0.25, 0.55
            else: w_n, w_l, w_t = 0.55, 0.15, 0.30
        
        final = (sent * w_n) + (tech * w_t) + (curr_lead * w_l)
        final = max(min(final, 1.0), -1.0)
        return round(50 + (final * 50), 2), round(tech, 2), round(sent, 2), round(curr_lead, 2)

# ==============================================================================
# 5. MAIN
# ==============================================================================

if __name__ == "__main__":
    scorer = HybridScorer()
    print(f"\n--- ANALISI CLUSTERIZZATA AVANZATA ({datetime.now().strftime('%Y-%m-%d')}) ---")
    [attachment_0](attachment)
    
    results = []
    leader_cache = {}

    # Costruzione lista lavoro
    WORK_LIST = []
    for raw in USER_SYMBOL_LIST:
        yf_tick = map_symbol_to_yahoo(raw)
        sec_name, bench_tick = get_sector_info(yf_tick, raw)
        WORK_LIST.append({"orig": raw, "yf": yf_tick, "sec": sec_name, "bench": bench_tick})

    # Ordinamento per settore per stampa pulita
    WORK_LIST.sort(key=lambda x: x['sec'])
    
    current_sector = ""
    
    for item in WORK_LIST:
        tick = item['yf']
        orig = item['orig']
        sec = item['sec']
        bench = item['bench']

        # Stampa intestazione settore se cambia
        if sec != current_sector:
            if bench not in leader_cache:
                leader_cache[bench] = get_leader_trend(bench)
            ld_score = leader_cache[bench]
            icon = "📈" if ld_score > 0 else "📉"
            print(f"\n📂 {sec} (Leader: {bench} {icon})")
            print("-" * 70)
            current_sector = sec
        else:
            ld_score = leader_cache[bench]

        # Processo Asset
        df = get_data(tick)
        if not df.empty:
            sentiment, count = get_news_data_advanced(tick, orig, SYMBOL_KEYWORDS)
            is_leader = (tick == bench)
            prob, tech, sent, lead = scorer.calculate_probability(df, sentiment, count, ld_score, is_leader)
            
            if prob >= 60: sig = "STRONG BUY"
            elif prob >= 53: sig = "BUY"
            elif prob <= 40: sig = "STRONG SELL"
            elif prob <= 47: sig = "SELL"
            else: sig = "HOLD"
            
            results.append({
                "Asset": orig, "Sector": sec, "Score": prob, "Signal": sig,
                "News": count, "Sent": sent, "Tech": tech, "Trend": lead
            })
            print(f"   {orig:<10} | {prob}% | {sig:<11} | News:{count}")
        else:
            print(f"   {orig:<10} | ⚠️ NO DATA")
        time.sleep(0.05)

    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by="Score", ascending=False)
        print("\n\n" + "="*100)
        print(f"🏆 CLASSIFICA FINALE (Top Picks)")
        print("="*100)
        print(f"{'ASSET':<10} | {'SECTOR':<15} | {'SCORE':<6} | {'SIGNAL':<11} | {'NEWS':<4} | {'SENT':<5} | {'TECH':<5} | {'LEAD':<5}")
        print("-" * 100)
        for _, row in df_res.iterrows():
            icon = "🟢" if "BUY" in row['Signal'] else "🔴" if "SELL" in row['Signal'] else "⚪"
            print(f"{row['Asset']:<10} | {row['Sector'][:15]:<15} | {row['Score']:<6} | {icon} {row['Signal']:<9} | {row['News']:<4} | {row['Sent']:<5} | {row['Tech']:<5} | {row['Trend']:<5}")
