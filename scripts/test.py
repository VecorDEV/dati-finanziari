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
# 1. NUOVE MAPPE DATI (Create su misura)
# ==============================================================================

# 1. MAPPA NOMI (Per ricerca News)
symbol_name_map = {
    # Stocks
    "AAPL": ["Apple", "Apple Inc."], "MSFT": ["Microsoft", "Microsoft Corporation"], "GOOGL": ["Google", "Alphabet"],
    "AMZN": ["Amazon", "Amazon.com"], "META": ["Meta", "Facebook"], "TSLA": ["Tesla", "Tesla Inc."],
    "V": ["Visa", "Visa Inc."], "JPM": ["JPMorgan", "JPMorgan Chase"], "JNJ": ["Johnson & Johnson"],
    "WMT": ["Walmart"], "NVDA": ["NVIDIA", "Nvidia Corp."], "PYPL": ["PayPal"], "DIS": ["Disney"],
    "NFLX": ["Netflix"], "NIO": ["NIO Inc."], "NRG": ["NRG Energy"], "ADBE": ["Adobe"],
    "INTC": ["Intel"], "CSCO": ["Cisco"], "PFE": ["Pfizer"], "KO": ["Coca-Cola"], "PEP": ["Pepsi"],
    "MRK": ["Merck"], "ABT": ["Abbott"], "XOM": ["ExxonMobil"], "CVX": ["Chevron"], "T": ["AT&T"],
    "MCD": ["McDonald's"], "NKE": ["Nike"], "HD": ["Home Depot"], "IBM": ["IBM"], "CRM": ["Salesforce"],
    "BMY": ["Bristol-Myers"], "ORCL": ["Oracle"], "ACN": ["Accenture"], "LLY": ["Eli Lilly"],
    "QCOM": ["Qualcomm"], "HON": ["Honeywell"], "COST": ["Costco"], "SBUX": ["Starbucks"],
    "CAT": ["Caterpillar"], "LOW": ["Lowe's"], "MS": ["Morgan Stanley"], "GS": ["Goldman Sachs"],
    "AXP": ["American Express"], "INTU": ["Intuit"], "AMGN": ["Amgen"], "GE": ["General Electric"],
    "FIS": ["Fidelity National"], "CVS": ["CVS Health"], "DE": ["Deere", "John Deere"],
    "BDX": ["Becton Dickinson"], "NOW": ["ServiceNow"], "SCHW": ["Charles Schwab"], "LMT": ["Lockheed Martin"],
    "ADP": ["ADP"], "C": ["Citigroup"], "PLD": ["Prologis"], "NSC": ["Norfolk Southern"],
    "TMUS": ["T-Mobile"], "ITW": ["Illinois Tool Works"], "FDX": ["FedEx"], "PNC": ["PNC Financial"],
    "SO": ["Southern Company"], "APD": ["Air Products"], "ADI": ["Analog Devices"], "ICE": ["Intercontinental Exchange"],
    "ZTS": ["Zoetis"], "TJX": ["TJX Companies"], "CL": ["Colgate-Palmolive"], "MMC": ["Marsh & McLennan"],
    "EL": ["Estée Lauder"], "GM": ["General Motors"], "CME": ["CME Group"], "EW": ["Edwards Lifesciences"],
    "AON": ["Aon"], "D": ["Dominion Energy"], "PSA": ["Public Storage"], "AEP": ["American Electric Power"],
    "TROW": ["T. Rowe Price"], "LNTH": ["Lantheus"], "HE": ["Hawaiian Electric"], "BTDR": ["Bitdeer"],
    "NAAS": ["NaaS Technology"], "SCHL": ["Scholastic"], "TGT": ["Target"], "SYK": ["Stryker"],
    "BKNG": ["Booking.com"], "DUK": ["Duke Energy"], "USB": ["U.S. Bancorp"], "BABA": ["Alibaba"],
    "HTZ": ["Hertz"], "UBER": ["Uber"], "LYFT": ["Lyft"], "PLTR": ["Palantir"], "SNOW": ["Snowflake"],
    "ROKU": ["Roku"], "TWLO": ["Twilio"], "SQ": ["Block", "Square"], "COIN": ["Coinbase"],
    "PST.MI": ["Poste Italiane"], "UCG.MI": ["Unicredit"], "ISP.MI": ["Intesa Sanpaolo"],
    "ENEL.MI": ["Enel"], "STLAM.MI": ["Stellantis"], "LDO.MI": ["Leonardo"], "RIVN": ["Rivian"],
    "LCID": ["Lucid Motors"], "DDOG": ["Datadog"], "NET": ["Cloudflare"], "SHOP": ["Shopify"],
    "ZI": ["ZoomInfo"], "ZM": ["Zoom Video"], "BIDU": ["Baidu"], "PDD": ["Pinduoduo"],
    "JD": ["JD.com"], "ARM": ["Arm Holdings"], "DUOL": ["Duolingo"], "PBR": ["Petrobras"],
    "VALE": ["Vale S.A."], "AMX": ["America Movil"],
    # Forex
    "EURUSD": ["EUR/USD"], "USDJPY": ["USD/JPY"], "GBPUSD": ["GBP/USD"], "AUDUSD": ["AUD/USD"],
    "USDCAD": ["USD/CAD"], "USDCHF": ["USD/CHF"], "NZDUSD": ["NZD/USD"], "EURGBP": ["EUR/GBP"],
    "EURJPY": ["EUR/JPY"], "GBPJPY": ["GBP/JPY"], "AUDJPY": ["AUD/JPY"], "CADJPY": ["CAD/JPY"],
    "CHFJPY": ["CHF/JPY"], "EURAUD": ["EUR/AUD"], "EURNZD": ["EUR/NZD"], "EURCAD": ["EUR/CAD"],
    "EURCHF": ["EUR/CHF"], "GBPCHF": ["GBP/CHF"], "AUDCAD": ["AUD/CAD"],
    # Index
    "SPX500": ["S&P 500"], "DJ30": ["Dow Jones"], "NAS100": ["Nasdaq 100"], "NASCOMP": ["Nasdaq Composite"],
    "RUS2000": ["Russell 2000"], "VIX": ["VIX Index"], "EU50": ["Euro Stoxx 50"], "ITA40": ["FTSE MIB"],
    "GER40": ["DAX 40"], "UK100": ["FTSE 100"], "FRA40": ["CAC 40"], "SWI20": ["SMI Index"],
    "ESP35": ["IBEX 35"], "NETH25": ["AEX Index"], "JPN225": ["Nikkei 225"], "HKG50": ["Hang Seng"],
    "CHN50": ["Shanghai Composite"], "IND50": ["Nifty 50"], "KOR200": ["KOSPI"],
    # Crypto
    "BTCUSD": ["Bitcoin"], "ETHUSD": ["Ethereum"], "LTCUSD": ["Litecoin"], "XRPUSD": ["Ripple"],
    "BCHUSD": ["Bitcoin Cash"], "EOSUSD": ["EOS"], "XLMUSD": ["Stellar"], "ADAUSD": ["Cardano"],
    "TRXUSD": ["Tron"], "NEOUSD": ["NEO"], "DASHUSD": ["Dash Coin"], "XMRUSD": ["Monero"],
    "ETCUSD": ["Ethereum Classic"], "ZECUSD": ["Zcash"], "BNBUSD": ["Binance Coin"], "DOGEUSD": ["Dogecoin"],
    "USDTUSD": ["Tether"], "LINKUSD": ["Chainlink"], "ATOMUSD": ["Cosmos"], "XTZUSD": ["Tezos"],
    # Commodities
    "COCOA": ["Cocoa Futures"], "GOLD": ["Gold Price"], "SILVER": ["Silver Price"],
    "OIL": ["Crude Oil"], "NATGAS": ["Natural Gas"]
}

# 2. MAPPA SETTORI (Assegnazione fissa)
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

# 3. LEADER DI SETTORE
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

# Genera la lista degli asset basandosi ESATTAMENTE sulle chiavi della mappa settoriale
USER_SYMBOL_LIST = list(asset_sector_map.keys())

# ==============================================================================
# 2. UTILITY E MAPPATURA YAHOO
# ==============================================================================

def map_symbol_to_yahoo(symbol):
    """Converte il tuo simbolo nel formato accettato da Yahoo Finance."""
    s = symbol.upper()
    
    # 1. Indici
    idx = {
        "SPX500":"^GSPC", "DJ30":"^DJI", "NAS100":"^NDX", "NASCOMP":"^IXIC",
        "RUS2000":"^RUT", "VIX":"^VIX", "EU50":"^STOXX50E", "ITA40":"FTSEMIB.MI",
        "GER40":"^GDAXI", "UK100":"^FTSE", "FRA40":"^FCHI", "SWI20":"^SSMI",
        "ESP35":"^IBEX", "NETH25":"^AEX", "JPN225":"^N225", "HKG50":"^HSI",
        "CHN50":"000001.SS", "IND50":"^NSEI", "KOR200":"^KS11"
    }
    if s in idx: return idx[s]

    # 2. Commodities (Futures)
    comm = {"GOLD":"GC=F", "SILVER":"SI=F", "OIL":"CL=F", "NATGAS":"NG=F", "COCOA":"CC=F"}
    if s in comm: return comm[s]

    # 3. Forex (es. EURUSD -> EURUSD=X)
    if len(s)==6 and s.isalpha() and ("USD" in s or "EUR" in s or "JPY" in s): 
        return f"{s}=X"

    # 4. Crypto (es. BTCUSD -> BTC-USD)
    if s.endswith("USD") and len(s)>3 and "USDCAD" not in s: 
        return s.replace("USD", "-USD")

    # 5. Default (Stocks: AAPL, ISP.MI, ecc.)
    return s 

def get_sector_and_leader(ticker_symbol):
    """Restituisce il settore e il ticker Yahoo del leader."""
    sector = asset_sector_map.get(ticker_symbol, "Unknown Sector")
    raw_leader = sector_leaders.get(sector, "SPX500") 
    yahoo_leader = map_symbol_to_yahoo(raw_leader)
    return sector, yahoo_leader

def get_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
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

def get_news_data_advanced(ticker_yahoo, original_symbol):
    """
    Ricerca news combinando il Ticker E i Nomi associati.
    Esempio per AAPL: Cerca ("AAPL" OR "Apple" OR "Apple Inc.") stock
    """
    # 1. Recupera i nomi estesi (es. ["Apple", "Apple Inc."])
    names = symbol_name_map.get(original_symbol, [])
    
    # 2. Crea la lista finale di ricerca: [Ticker] + [Nomi]
    search_terms = [original_symbol] + names
    
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
# 3. ENGINE
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
# 4. MAIN
# ==============================================================================

if __name__ == "__main__":
    scorer = HybridScorer()
    print(f"\n--- ANALISI PORTAFOGLIO STRATEGICA ({datetime.now().strftime('%Y-%m-%d')}) ---")
    
    results = []
    leader_cache = {}

    WORK_LIST = []
    for raw in USER_SYMBOL_LIST:
        yf_tick = map_symbol_to_yahoo(raw)
        sec_name, leader_yf_tick = get_sector_and_leader(raw)
        WORK_LIST.append({"orig": raw, "yf": yf_tick, "sec": sec_name, "bench": leader_yf_tick})

    WORK_LIST.sort(key=lambda x: x['sec'])
    
    current_sector = ""
    
    for item in WORK_LIST:
        tick = item['yf']
        orig = item['orig']
        sec = item['sec']
        bench = item['bench']

        if sec != current_sector:
            if bench not in leader_cache:
                leader_cache[bench] = get_leader_trend(bench)
            ld_score = leader_cache[bench]
            icon = "📈" if ld_score > 0 else "📉"
            print(f"\n📂 {sec} (Leader: {bench} {icon})")
            print("-" * 75)
            current_sector = sec
        else:
            ld_score = leader_cache[bench]

        df = get_data(tick)
        if not df.empty:
            sentiment, count = get_news_data_advanced(tick, orig) 
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
        print(f"{'ASSET':<10} | {'SECTOR':<20} | {'SCORE':<6} | {'SIGNAL':<11} | {'NEWS':<4} | {'SENT':<5} | {'TECH':<5} | {'LEAD':<5}")
        print("-" * 100)
        for _, row in df_res.iterrows():
            icon = "🟢" if "BUY" in row['Signal'] else "🔴" if "SELL" in row['Signal'] else "⚪"
            print(f"{row['Asset']:<10} | {row['Sector'][:20]:<20} | {row['Score']:<6} | {icon} {row['Signal']:<9} | {row['News']:<4} | {row['Sent']:<5} | {row['Tech']:<5} | {row['Trend']:<5}")
