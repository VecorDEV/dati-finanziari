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
# 1. IL CUORE DEL SISTEMA: LA MAPPA DI CONVERSIONE
# ==============================================================================
# Chiave: Il nome che vuoi vedere tu (e usare per i settori/news)
# Valore: Il codice tecnico che vuole Yahoo Finance
TICKER_MAP = {
    # Stocks US (Standard)
    "AAPL": "AAPL", "MSFT": "MSFT", "GOOGL": "GOOGL", "AMZN": "AMZN", "META": "META",
    "TSLA": "TSLA", "V": "V", "JPM": "JPM", "JNJ": "JNJ", "WMT": "WMT",
    "NVDA": "NVDA", "PYPL": "PYPL", "DIS": "DIS", "NFLX": "NFLX", "NIO": "NIO",
    "NRG": "NRG", "ADBE": "ADBE", "INTC": "INTC", "CSCO": "CSCO", "PFE": "PFE",
    "KO": "KO", "PEP": "PEP", "MRK": "MRK", "ABT": "ABT", "XOM": "XOM",
    "CVX": "CVX", "T": "T", "MCD": "MCD", "NKE": "NKE", "HD": "HD",
    "IBM": "IBM", "CRM": "CRM", "BMY": "BMY", "ORCL": "ORCL", "ACN": "ACN",
    "LLY": "LLY", "QCOM": "QCOM", "HON": "HON", "COST": "COST", "SBUX": "SBUX",
    "CAT": "CAT", "LOW": "LOW", "MS": "MS", "GS": "GS", "AXP": "AXP",
    "INTU": "INTU", "AMGN": "AMGN", "GE": "GE", "FIS": "FIS", "CVS": "CVS",
    "DE": "DE", "BDX": "BDX", "NOW": "NOW", "SCHW": "SCHW", "LMT": "LMT",
    "ADP": "ADP", "C": "C", "PLD": "PLD", "NSC": "NSC", "TMUS": "TMUS",
    "ITW": "ITW", "FDX": "FDX", "PNC": "PNC", "SO": "SO", "APD": "APD",
    "ADI": "ADI", "ICE": "ICE", "ZTS": "ZTS", "TJX": "TJX", "CL": "CL",
    "MMC": "MMC", "EL": "EL", "GM": "GM", "CME": "CME", "EW": "EW",
    "AON": "AON", "D": "D", "PSA": "PSA", "AEP": "AEP", "TROW": "TROW",
    "LNTH": "LNTH", "HE": "HE", "BTDR": "BTDR", "NAAS": "NAAS", "SCHL": "SCHL",
    "TGT": "TGT", "SYK": "SYK", "BKNG": "BKNG", "DUK": "DUK", "USB": "USB",
    
    # Stocks Growth / Tech / New
    "ARM": "ARM", "BABA": "BABA", "BIDU": "BIDU", "COIN": "COIN", "DDOG": "DDOG", 
    "HTZ": "HTZ", "JD": "JD", "LCID": "LCID", "LYFT": "LYFT", "NET": "NET", 
    "PDD": "PDD", "PLTR": "PLTR", "RIVN": "RIVN", "ROKU": "ROKU", "SHOP": "SHOP", 
    "SNOW": "SNOW", "SQ": "SQ", "TWLO": "TWLO", "UBER": "UBER", "ZI": "ZI", 
    "ZM": "ZM", "DUOL": "DUOL", "PBR": "PBR", "VALE": "VALE", "AMX": "AMX",

    # Stocks Europa / Milano
    "ISP.MI": "ISP.MI", "ENEL.MI": "ENEL.MI", "STLAM.MI": "STLAM.MI", 
    "LDO.MI": "LDO.MI", "PST.MI": "PST.MI", "UCG.MI": "UCG.MI",

    # Forex (=X)
    "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X", "GBPUSD": "GBPUSD=X", 
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", 
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X", 
    "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "CADJPY": "CADJPY=X", 
    "CHFJPY": "CHFJPY=X", "EURAUD": "EURAUD=X", "EURNZD": "EURNZD=X", 
    "EURCAD": "EURCAD=X", "EURCHF": "EURCHF=X", "GBPCHF": "GBPCHF=X", 
    "AUDCAD": "AUDCAD=X",

    # Indici Globali (Ticker complessi)
    "SPX500": "^GSPC", "DJ30": "^DJI", "NAS100": "^NDX", "NASCOMP": "^IXIC", 
    "RUS2000": "^RUT", "VIX": "^VIX", "EU50": "^STOXX50E", "ITA40": "FTSEMIB.MI", 
    "GER40": "^GDAXI", "UK100": "^FTSE", "FRA40": "^FCHI", "SWI20": "^SSMI", 
    "ESP35": "^IBEX", "NETH25": "^AEX", "JPN225": "^N225", "HKG50": "^HSI", 
    "CHN50": "000001.SS", "IND50": "^NSEI", "KOR200": "^KS11",

    # Crypto (-USD)
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "LTCUSD": "LTC-USD", 
    "XRPUSD": "XRP-USD", "BCHUSD": "BCH-USD", "EOSUSD": "EOS-USD", 
    "XLMUSD": "XLM-USD", "ADAUSD": "ADA-USD", "TRXUSD": "TRX-USD", 
    "NEOUSD": "NEO-USD", "DASHUSD": "DASH-USD", "XMRUSD": "XMR-USD", 
    "ETCUSD": "ETC-USD", "ZECUSD": "ZEC-USD", "BNBUSD": "BNB-USD", 
    "DOGEUSD": "DOGE-USD", "USDTUSD": "USDT-USD", "LINKUSD": "LINK-USD", 
    "ATOMUSD": "ATOM-USD", "XTZUSD": "XTZ-USD",

    # Commodities (Futures)
    "COCOA": "CC=F", "GOLD": "GC=F", "SILVER": "SI=F", "OIL": "CL=F", "NATGAS": "NG=F"
}

# ==============================================================================
# 2. METADATI (Nomi Estesi e Settori)
# ==============================================================================

symbol_name_map = {
    # Stocks
    "AAPL": ["Apple", "Apple Inc."], "MSFT": ["Microsoft"], "GOOGL": ["Google", "Alphabet"],
    "AMZN": ["Amazon"], "META": ["Meta", "Facebook"], "TSLA": ["Tesla"],
    "NVDA": ["NVIDIA"], "AMD": ["AMD"], "NFLX": ["Netflix"], "DIS": ["Disney"],
    "GOLD": ["Gold Price", "XAUUSD"], "OIL": ["Crude Oil", "WTI"], 
    "BTCUSD": ["Bitcoin"], "ETHUSD": ["Ethereum"], "SPX500": ["S&P 500", "US Stock Market"],
    # ... (Puoi lasciare la tua lista completa qui, il codice userà .get() quindi è sicuro)
    "PST.MI": ["Poste Italiane"], "UCG.MI": ["Unicredit"], "ISP.MI": ["Intesa Sanpaolo"],
    "ENEL.MI": ["Enel"], "STLAM.MI": ["Stellantis"], "LDO.MI": ["Leonardo"]
}

asset_sector_map = {
    # 1. Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "ADBE": "Technology",
    "INTC": "Technology", "CSCO": "Technology", "IBM": "Technology", "CRM": "Technology",
    "ORCL": "Technology", "ACN": "Technology", "QCOM": "Technology", "INTU": "Technology",
    "NOW": "Technology", "ADI": "Technology", "DDOG": "Technology", "NET": "Technology",
    "SHOP": "Technology", "ZI": "Technology", "ZM": "Technology", "ARM": "Technology",
    "PLTR": "Technology", "SNOW": "Technology", "TWLO": "Technology", "BTDR": "Technology",
    # 2. Communication Services
    "GOOGL": "Communication Services", "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "T": "Communication Services", "TMUS": "Communication Services",
    "BIDU": "Communication Services", "ROKU": "Communication Services", "AMX": "Communication Services",
    "SCHL": "Communication Services",
    # 3. Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "HD": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TJX": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "BABA": "Consumer Discretionary", "JD": "Consumer Discretionary",
    "PDD": "Consumer Discretionary", "NIO": "Consumer Discretionary", "RIVN": "Consumer Discretionary",
    "LCID": "Consumer Discretionary", "HTZ": "Consumer Discretionary", "UBER": "Consumer Discretionary",
    "LYFT": "Consumer Discretionary", "STLAM.MI": "Consumer Discretionary", "DUOL": "Consumer Discretionary",
    "NAAS": "Consumer Discretionary",
    # 4. Consumer Staples
    "WMT": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "CL": "Consumer Staples", "EL": "Consumer Staples",
    "TGT": "Consumer Staples", "COCOA": "Consumer Staples",
    # 5. Financials
    "JPM": "Financials", "V": "Financials", "PYPL": "Financials", "MS": "Financials",
    "GS": "Financials", "AXP": "Financials", "C": "Financials", "SCHW": "Financials",
    "FIS": "Financials", "MMC": "Financials", "AON": "Financials", "TROW": "Financials",
    "CME": "Financials", "ICE": "Financials", "USB": "Financials", "SQ": "Financials",
    "COIN": "Financials", "PST.MI": "Financials", "UCG.MI": "Financials", "ISP.MI": "Financials",
    "PNC": "Financials",
    # 6. Health Care
    "JNJ": "Health Care", "PFE": "Health Care", "MRK": "Health Care", "ABT": "Health Care",
    "BMY": "Health Care", "LLY": "Health Care", "AMGN": "Health Care", "CVS": "Health Care",
    "BDX": "Health Care", "ZTS": "Health Care", "EW": "Health Care", "SYK": "Health Care",
    "LNTH": "Health Care",
    # 7. Energy
    "XOM": "Energy", "CVX": "Energy", "PBR": "Energy", "OIL": "Energy", "NATGAS": "Energy",
    # 8. Industrials
    "HON": "Industrials", "CAT": "Industrials", "GE": "Industrials", "LMT": "Industrials",
    "FDX": "Industrials", "NSC": "Industrials", "ITW": "Industrials", "DE": "Industrials",
    "ADP": "Industrials", "LDO.MI": "Industrials",
    # 9. Materials
    "APD": "Materials", "VALE": "Materials", "GOLD": "Materials", "SILVER": "Materials",
    # 10. Utilities
    "NRG": "Utilities", "SO": "Utilities", "D": "Utilities", "AEP": "Utilities",
    "HE": "Utilities", "DUK": "Utilities", "ENEL.MI": "Utilities",
    # 11. Real Estate
    "PLD": "Real Estate", "PSA": "Real Estate",
    # 12. Crypto & Digital Assets
    "BTCUSD": "Crypto & Digital Assets", "ETHUSD": "Crypto & Digital Assets",
    "LTCUSD": "Crypto & Digital Assets", "XRPUSD": "Crypto & Digital Assets",
    "BCHUSD": "Crypto & Digital Assets", "EOSUSD": "Crypto & Digital Assets",
    "XLMUSD": "Crypto & Digital Assets", "ADAUSD": "Crypto & Digital Assets",
    "TRXUSD": "Crypto & Digital Assets", "NEOUSD": "Crypto & Digital Assets",
    "DASHUSD": "Crypto & Digital Assets", "XMRUSD": "Crypto & Digital Assets",
    "ETCUSD": "Crypto & Digital Assets", "ZECUSD": "Crypto & Digital Assets",
    "BNBUSD": "Crypto & Digital Assets", "DOGEUSD": "Crypto & Digital Assets",
    "USDTUSD": "Crypto & Digital Assets", "LINKUSD": "Crypto & Digital Assets",
    "ATOMUSD": "Crypto & Digital Assets", "XTZUSD": "Crypto & Digital Assets",
    # 13. Indices/ETF/Cash (Forex incluso)
    "SPX500": "Indices/ETF/Cash", "DJ30": "Indices/ETF/Cash", "NAS100": "Indices/ETF/Cash",
    "NASCOMP": "Indices/ETF/Cash", "RUS2000": "Indices/ETF/Cash", "VIX": "Indices/ETF/Cash",
    "EU50": "Indices/ETF/Cash", "ITA40": "Indices/ETF/Cash", "GER40": "Indices/ETF/Cash",
    "UK100": "Indices/ETF/Cash", "FRA40": "Indices/ETF/Cash", "SWI20": "Indices/ETF/Cash",
    "ESP35": "Indices/ETF/Cash", "NETH25": "Indices/ETF/Cash", "JPN225": "Indices/ETF/Cash",
    "HKG50": "Indices/ETF/Cash", "CHN50": "Indices/ETF/Cash", "IND50": "Indices/ETF/Cash",
    "KOR200": "Indices/ETF/Cash",
    "EURUSD": "Indices/ETF/Cash", "USDJPY": "Indices/ETF/Cash", "GBPUSD": "Indices/ETF/Cash",
    "AUDUSD": "Indices/ETF/Cash", "USDCAD": "Indices/ETF/Cash", "USDCHF": "Indices/ETF/Cash",
    "NZDUSD": "Indices/ETF/Cash", "EURGBP": "Indices/ETF/Cash", "EURJPY": "Indices/ETF/Cash",
    "GBPJPY": "Indices/ETF/Cash", "AUDJPY": "Indices/ETF/Cash", "CADJPY": "Indices/ETF/Cash",
    "CHFJPY": "Indices/ETF/Cash", "EURAUD": "Indices/ETF/Cash", "EURNZD": "Indices/ETF/Cash",
    "EURCAD": "Indices/ETF/Cash", "EURCHF": "Indices/ETF/Cash", "GBPCHF": "Indices/ETF/Cash",
    "AUDCAD": "Indices/ETF/Cash"
}

sector_leaders = {
    "Technology": "AAPL",             
    "Communication Services": "GOOGL",
    "Consumer Discretionary": "AMZN", 
    "Consumer Staples": "WMT",        
    "Financials": "JPM",              
    "Health Care": "LLY",             
    "Energy": "XOM",                  
    "Industrials": "CAT",             
    "Materials": "GOLD",              
    "Utilities": "SO",                
    "Real Estate": "PLD",             
    "Crypto & Digital Assets": "BTCUSD",
    "Indices/ETF/Cash": "SPX500"      
}

# La lista su cui lavorare sono le chiavi della mappa settoriale
# (Queste sono le tue stringhe "User Friendly")
USER_SYMBOL_LIST = list(asset_sector_map.keys())

# ==============================================================================
# 3. UTILITY FUNZIONI
# ==============================================================================

def get_sector_and_leader(friendly_symbol):
    """
    Trova il settore e il ticker Yahoo del leader di quel settore.
    """
    # 1. Trova il settore
    sector = asset_sector_map.get(friendly_symbol, "Unknown Sector")
    
    # 2. Trova chi è il leader "friendly" (es. 'GOLD')
    friendly_leader = sector_leaders.get(sector, "SPX500") 
    
    # 3. Converti il leader friendly in Yahoo Ticker (es. 'GOLD' -> 'GC=F')
    # Se non c'è nella mappa, usa il friendly name come fallback
    yahoo_leader = TICKER_MAP.get(friendly_leader, friendly_leader)
    
    return sector, yahoo_leader

def get_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        
        # Gestione MultiIndex (yfinance recente a volte ritorna colonne doppie)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # Se c'è solo un ticker, droppa il livello del ticker
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

def get_news_data_advanced(ticker_yahoo, friendly_symbol):
    """
    ticker_yahoo: serve per coerenza tecnica (non usato nella query qui sotto ma utile se volessi estenderlo)
    friendly_symbol: usato per cercare i nomi "umani" nel dizionario
    """
    # 1. Recupera i nomi estesi (es. ["Apple", "Apple Inc."])
    names = symbol_name_map.get(friendly_symbol, [])
    
    # 2. Crea la lista finale di ricerca: [Simbolo Amichevole] + [Nomi Estesi]
    search_terms = [friendly_symbol] + names
    
    # 3. Costruisce la query limitandosi ai primi 3 termini
    query_items = [f'"{term}"' for term in search_terms[:3]]
    final_query = " OR ".join(query_items)
    
    rss_url = f"https://news.google.com/rss/search?q=({final_query})+stock&hl=en-US&gl=US&ceid=US:en"
    
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
            'surge': 4.0, 'jump': 2.0, 'rally': 3.5, 'soar': 4.0, 'bull': 3.0, 
            'plunge': -4.0, 'crash': -4.0, 'drop': -3.0, 'record': 2.0, 'high': 1.5
        }
        sia.lexicon.update(lexicon)
        
        total = sum([sia.polarity_scores(t)['compound'] for t in titles])
        return (total / count), count
        
    except Exception as e:
        return 0.0, 0

# ==============================================================================
# 4. ENGINE DI CALCOLO
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
# 5. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    scorer = HybridScorer()
    print(f"\n--- ANALISI PORTAFOGLIO STRATEGICA ({datetime.now().strftime('%Y-%m-%d')}) ---")
    
    results = []
    leader_cache = {}

    WORK_LIST = []
    
    # Costruiamo la lista di lavoro unendo:
    # 1. Il nome friendly (da usare per print e news)
    # 2. Il ticker Yahoo (da usare per scaricare i dati)
    # 3. Il settore e il ticker del leader
    
    for friendly_name in USER_SYMBOL_LIST:
        # Recupera il ticker Yahoo dal dizionario. Se non c'è, usa il friendly name (fallback)
        yahoo_ticker = TICKER_MAP.get(friendly_name, friendly_name)
        
        sec_name, leader_yf_tick = get_sector_and_leader(friendly_name)
        
        WORK_LIST.append({
            "friendly": friendly_name,
            "yahoo": yahoo_ticker,
            "sec": sec_name,
            "bench": leader_yf_tick
        })

    # Ordina per settore per una stampa pulita
    WORK_LIST.sort(key=lambda x: x['sec'])
    
    current_sector = ""
    
    for item in WORK_LIST:
        friendly = item['friendly']
        yahoo_t = item['yahoo']
        sec = item['sec']
        bench = item['bench']

        # Gestione cambio settore e calcolo Trend Leader
        if sec != current_sector:
            if bench not in leader_cache:
                leader_cache[bench] = get_leader_trend(bench)
            ld_score = leader_cache[bench]
            icon = "📈" if ld_score > 0 else "📉"
            # Per mostrare il nome del leader, cerchiamo di invertire la mappa o usare il ticker
            print(f"\n📂 {sec} (Leader Ticker: {bench} {icon})")
            print("-" * 75)
            current_sector = sec
        else:
            ld_score = leader_cache[bench]

        # Scarica dati usando il Ticker Yahoo
        df = get_data(yahoo_t)
        
        if not df.empty:
            # Cerca news usando il nome friendly
            sentiment, count = get_news_data_advanced(yahoo_t, friendly) 
            
            is_leader = (yahoo_t == bench)
            prob, tech, sent, lead = scorer.calculate_probability(df, sentiment, count, ld_score, is_leader)
            
            if prob >= 60: sig = "STRONG BUY"
            elif prob >= 53: sig = "BUY"
            elif prob <= 40: sig = "STRONG SELL"
            elif prob <= 47: sig = "SELL"
            else: sig = "HOLD"
            
            results.append({
                "Asset": friendly, "Sector": sec, "Score": prob, "Signal": sig,
                "News": count, "Sent": sent, "Tech": tech, "Trend": lead
            })
            # Stampa usando il nome Friendly
            print(f"   {friendly:<10} | {prob}% | {sig:<11} | News:{count}")
        else:
            print(f"   {friendly:<10} | ⚠️ NO DATA (Yahoo Ticker: {yahoo_t})")
        
        # Rispetto per le API
        time.sleep(0.05)

    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by="Score", ascending=False)
        print("\n\n" + "="*100)
        print(f"🏆 CLASSIFICA FINALE (Top Picks)")
        print("="*100)
        print(f"{'ASSET':<10} | {'SECTOR':<20} | {'SCORE':<6} | {'SIGNAL':<11} | {'NEWS':<4} | {'SENT':<5} | {'TECH':<5} | {'LEAD':<5}")
        print("-" * 100)
        for _, row in df_res.iterrows():
            icon = "🟢" if "BUY" in row['Signal'] else "🔴" if "SELL" in row['Signal'] else "⚪"
            print(f"{row['Asset']:<10} | {row['Sector'][:20]:<20} | {row['Score']:<6} | {icon} {row['Signal']:<9} | {row['News']:<4} | {row['Sent']:<5} | {row['Tech']:<5} | {row['Trend']:<5}")
