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
# 1. MAPPA TICKER (Logica Tecnica)
# ==============================================================================
TICKER_MAP = {
    # --- US Stocks ---
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
    "CHTR": "CHTR", 

    # --- Growth / New ---
    "ARM": "ARM", "BABA": "BABA", "BIDU": "BIDU", "COIN": "COIN", "DDOG": "DDOG", 
    "HTZ": "HTZ", "JD": "JD", "LCID": "LCID", "LYFT": "LYFT", "NET": "NET", 
    "PDD": "PDD", "PLTR": "PLTR", "RIVN": "RIVN", "ROKU": "ROKU", "SHOP": "SHOP", 
    "SNOW": "SNOW", "SQ": "SQ", "TWLO": "TWLO", "UBER": "UBER", "ZI": "ZI", 
    "ZM": "ZM", "DUOL": "DUOL", "PBR": "PBR", "VALE": "VALE", "AMX": "AMX",

    # --- Europe / Italy ---
    "ISP.MI": "ISP.MI", "ENEL.MI": "ENEL.MI", "STLAM.MI": "STLAM.MI", 
    "LDO.MI": "LDO.MI", "PST.MI": "PST.MI", "UCG.MI": "UCG.MI",

    # --- Forex ---
    "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X", "GBPUSD": "GBPUSD=X", 
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", 
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X", 
    "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "CADJPY": "CADJPY=X", 
    "CHFJPY": "CHFJPY=X", "EURAUD": "EURAUD=X", "EURNZD": "EURNZD=X", 
    "EURCAD": "EURCAD=X", "EURCHF": "EURCHF=X", "GBPCHF": "GBPCHF=X", 
    "AUDCAD": "AUDCAD=X",

    # --- Indices ---
    "SPX500": "^GSPC", "DJ30": "^DJI", "NAS100": "^NDX", "NASCOMP": "^IXIC", 
    "RUS2000": "^RUT", "VIX": "^VIX", "EU50": "^STOXX50E", "ITA40": "FTSEMIB.MI", 
    "GER40": "^GDAXI", "UK100": "^FTSE", "FRA40": "^FCHI", "SWI20": "^SSMI", 
    "ESP35": "^IBEX", "NETH25": "^AEX", "JPN225": "^N225", "HKG50": "^HSI", 
    "CHN50": "000001.SS", "IND50": "^NSEI", "KOR200": "^KS11",

    # --- Crypto ---
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "LTCUSD": "LTC-USD", 
    "XRPUSD": "XRP-USD", "BCHUSD": "BCH-USD", "EOSUSD": "EOS-USD", 
    "XLMUSD": "XLM-USD", "ADAUSD": "ADA-USD", "TRXUSD": "TRX-USD", 
    "NEOUSD": "NEO-USD", "DASHUSD": "DASH-USD", "XMRUSD": "XMR-USD", 
    "ETCUSD": "ETC-USD", "ZECUSD": "ZEC-USD", "BNBUSD": "BNB-USD", 
    "DOGEUSD": "DOGE-USD", "USDTUSD": "USDT-USD", "LINKUSD": "LINK-USD", 
    "ATOMUSD": "ATOM-USD", "XTZUSD": "XTZ-USD",

    # --- Commodities ---
    "COCOA": "CC=F", "GOLD": "GC=F", "SILVER": "SI=F", "OIL": "CL=F", "NATGAS": "NG=F"
}

# ==============================================================================
# 2. SETTORI E LEADERS
# ==============================================================================

sector_leaders = {
    "1. Big Tech, Software & Internet": "MSFT",
    "2. Semiconductors & AI": "NVDA",
    "3. Financial Services": "JPM",
    "4. Automotive & Mobility": "TSLA",
    "5. Healthcare & Pharma": "LLY",
    "6. Consumer Goods & Retail": "WMT",
    "7. Industrials & Defense": "CAT",
    "8. Energy (Oil & Gas)": "OIL",
    "9. Utilities & Green": "SO",
    "10. Precious Metals & Materials": "GOLD",
    "11. Media & Telecom": "NFLX",
    "12. Indices (Global)": "SPX500",
    "13. Forex (Currencies)": "EURUSD",
    "14. Crypto Assets": "BTCUSD"
}

asset_sector_map = {
    # 1. Tech
    "AAPL": "1. Big Tech, Software & Internet", "MSFT": "1. Big Tech, Software & Internet", 
    "GOOGL": "1. Big Tech, Software & Internet", "AMZN": "1. Big Tech, Software & Internet",
    "META": "1. Big Tech, Software & Internet", "ADBE": "1. Big Tech, Software & Internet",
    "CRM": "1. Big Tech, Software & Internet", "ORCL": "1. Big Tech, Software & Internet",
    "IBM": "1. Big Tech, Software & Internet", "NOW": "1. Big Tech, Software & Internet",
    "INTU": "1. Big Tech, Software & Internet", "ADP": "1. Big Tech, Software & Internet",
    "BABA": "1. Big Tech, Software & Internet", "BIDU": "1. Big Tech, Software & Internet",
    "SHOP": "1. Big Tech, Software & Internet", "SNOW": "1. Big Tech, Software & Internet",
    "PLTR": "1. Big Tech, Software & Internet", "TWLO": "1. Big Tech, Software & Internet",
    "ZI": "1. Big Tech, Software & Internet", "ZM": "1. Big Tech, Software & Internet",
    "DUOL": "1. Big Tech, Software & Internet", "JD": "1. Big Tech, Software & Internet",
    "NET": "1. Big Tech, Software & Internet", "PDD": "1. Big Tech, Software & Internet",
    "BTDR": "1. Big Tech, Software & Internet", "DDOG": "1. Big Tech, Software & Internet",
    # 2. Semi
    "NVDA": "2. Semiconductors & AI", "INTC": "2. Semiconductors & AI",
    "QCOM": "2. Semiconductors & AI", "ADI": "2. Semiconductors & AI",
    "ARM": "2. Semiconductors & AI", "CSCO": "2. Semiconductors & AI",
    "ACN": "2. Semiconductors & AI", "FIS": "2. Semiconductors & AI",
    # 3. Fin
    "JPM": "3. Financial Services", "V": "3. Financial Services", 
    "PYPL": "3. Financial Services", "MS": "3. Financial Services",
    "GS": "3. Financial Services", "AXP": "3. Financial Services",
    "SCHW": "3. Financial Services", "C": "3. Financial Services",
    "PLD": "3. Financial Services", "PNC": "3. Financial Services",
    "ICE": "3. Financial Services", "MMC": "3. Financial Services",
    "CME": "3. Financial Services", "AON": "3. Financial Services",
    "TROW": "3. Financial Services", "USB": "3. Financial Services",
    "PSA": "3. Financial Services", "COIN": "3. Financial Services",
    "SQ": "3. Financial Services", "ISP.MI": "3. Financial Services",
    "UCG.MI": "3. Financial Services", "PST.MI": "3. Financial Services",
    # 4. Auto
    "TSLA": "4. Automotive & Mobility", "GM": "4. Automotive & Mobility",
    "NIO": "4. Automotive & Mobility", "STLAM.MI": "4. Automotive & Mobility",
    "HTZ": "4. Automotive & Mobility", "LCID": "4. Automotive & Mobility",
    "RIVN": "4. Automotive & Mobility", "UBER": "4. Automotive & Mobility",
    "LYFT": "4. Automotive & Mobility", "NAAS": "4. Automotive & Mobility",
    # 5. Health
    "LLY": "5. Healthcare & Pharma", "JNJ": "5. Healthcare & Pharma",
    "PFE": "5. Healthcare & Pharma", "MRK": "5. Healthcare & Pharma",
    "ABT": "5. Healthcare & Pharma", "BMY": "5. Healthcare & Pharma",
    "AMGN": "5. Healthcare & Pharma", "CVS": "5. Healthcare & Pharma",
    "BDX": "5. Healthcare & Pharma", "ZTS": "5. Healthcare & Pharma",
    "EW": "5. Healthcare & Pharma", "LNTH": "5. Healthcare & Pharma",
    "SYK": "5. Healthcare & Pharma",
    # 6. Consumer
    "WMT": "6. Consumer Goods & Retail", "KO": "6. Consumer Goods & Retail",
    "PEP": "6. Consumer Goods & Retail", "MCD": "6. Consumer Goods & Retail",
    "NKE": "6. Consumer Goods & Retail", "HD": "6. Consumer Goods & Retail",
    "COST": "6. Consumer Goods & Retail", "SBUX": "6. Consumer Goods & Retail",
    "LOW": "6. Consumer Goods & Retail", "TGT": "6. Consumer Goods & Retail",
    "TJX": "6. Consumer Goods & Retail", "CL": "6. Consumer Goods & Retail",
    "EL": "6. Consumer Goods & Retail", "SCHL": "6. Consumer Goods & Retail",
    "COCOA": "6. Consumer Goods & Retail",
    # 7. Ind
    "CAT": "7. Industrials & Defense", "LMT": "7. Industrials & Defense",
    "ITW": "7. Industrials & Defense", "FDX": "7. Industrials & Defense",
    "NSC": "7. Industrials & Defense", "GE": "7. Industrials & Defense",
    "HON": "7. Industrials & Defense", "DE": "7. Industrials & Defense",
    "LDO.MI": "7. Industrials & Defense", "BKNG": "7. Industrials & Defense",
    # 8. Energy
    "OIL": "8. Energy (Oil & Gas)", "NATGAS": "8. Energy (Oil & Gas)",
    "XOM": "8. Energy (Oil & Gas)", "CVX": "8. Energy (Oil & Gas)",
    "PBR": "8. Energy (Oil & Gas)", "NRG": "8. Energy (Oil & Gas)",
    # 9. Util
    "SO": "9. Utilities & Green", "ENEL.MI": "9. Utilities & Green",
    "DUK": "9. Utilities & Green", "AEP": "9. Utilities & Green",
    "D": "9. Utilities & Green", "HE": "9. Utilities & Green",
    "APD": "9. Utilities & Green",
    # 10. Metal
    "GOLD": "10. Precious Metals & Materials", "SILVER": "10. Precious Metals & Materials",
    "VALE": "10. Precious Metals & Materials",
    # 11. Media
    "NFLX": "11. Media & Telecom", "DIS": "11. Media & Telecom",
    "T": "11. Media & Telecom", "TMUS": "11. Media & Telecom",
    "AMX": "11. Media & Telecom", "ROKU": "11. Media & Telecom",
    "CHTR": "11. Media & Telecom", 
    # 12. Indices
    "SPX500": "12. Indices (Global)", "DJ30": "12. Indices (Global)",
    "NAS100": "12. Indices (Global)", "NASCOMP": "12. Indices (Global)",
    "RUS2000": "12. Indices (Global)", "VIX": "12. Indices (Global)",
    "EU50": "12. Indices (Global)", "ITA40": "12. Indices (Global)",
    "GER40": "12. Indices (Global)", "UK100": "12. Indices (Global)",
    "FRA40": "12. Indices (Global)", "SWI20": "12. Indices (Global)",
    "ESP35": "12. Indices (Global)", "NETH25": "12. Indices (Global)",
    "JPN225": "12. Indices (Global)", "HKG50": "12. Indices (Global)",
    "CHN50": "12. Indices (Global)", "IND50": "12. Indices (Global)",
    "KOR200": "12. Indices (Global)",
    # 13. Forex
    "EURUSD": "13. Forex (Currencies)", "USDJPY": "13. Forex (Currencies)",
    "GBPUSD": "13. Forex (Currencies)", "AUDUSD": "13. Forex (Currencies)",
    "USDCAD": "13. Forex (Currencies)", "USDCHF": "13. Forex (Currencies)",
    "NZDUSD": "13. Forex (Currencies)", "EURGBP": "13. Forex (Currencies)",
    "EURJPY": "13. Forex (Currencies)", "GBPJPY": "13. Forex (Currencies)",
    "AUDJPY": "13. Forex (Currencies)", "CADJPY": "13. Forex (Currencies)",
    "CHFJPY": "13. Forex (Currencies)", "EURAUD": "13. Forex (Currencies)",
    "EURNZD": "13. Forex (Currencies)", "EURCAD": "13. Forex (Currencies)",
    "EURCHF": "13. Forex (Currencies)", "GBPCHF": "13. Forex (Currencies)",
    "AUDCAD": "13. Forex (Currencies)",
    # 14. Crypto
    "BTCUSD": "14. Crypto Assets", "ETHUSD": "14. Crypto Assets",
    "LTCUSD": "14. Crypto Assets", "XRPUSD": "14. Crypto Assets",
    "BCHUSD": "14. Crypto Assets", "EOSUSD": "14. Crypto Assets",
    "XLMUSD": "14. Crypto Assets", "ADAUSD": "14. Crypto Assets",
    "TRXUSD": "14. Crypto Assets", "NEOUSD": "14. Crypto Assets",
    "DASHUSD": "14. Crypto Assets", "XMRUSD": "14. Crypto Assets",
    "ETCUSD": "14. Crypto Assets", "ZECUSD": "14. Crypto Assets",
    "BNBUSD": "14. Crypto Assets", "DOGEUSD": "14. Crypto Assets",
    "USDTUSD": "14. Crypto Assets", "LINKUSD": "14. Crypto Assets",
    "ATOMUSD": "14. Crypto Assets", "XTZUSD": "14. Crypto Assets",
}

# MAPPA NOMI ESTESI (COMPLETA E DETTAGLIATA)
symbol_name_map = {
    # Stocks
    "AAPL": ["Apple", "Apple Inc."],
    "MSFT": ["Microsoft", "Microsoft Corporation"],
    "GOOGL": ["Google", "Alphabet", "Alphabet Inc."],
    "AMZN": ["Amazon", "Amazon.com"],
    "META": ["Meta", "Facebook", "Meta Platforms"],
    "TSLA": ["Tesla", "Tesla Inc."],
    "V": ["Visa", "Visa Inc."],
    "JPM": ["JPMorgan", "JPMorgan Chase"],
    "JNJ": ["Johnson & Johnson", "JNJ"],
    "WMT": ["Walmart"],
    "NVDA": ["NVIDIA", "Nvidia Corp."],
    "PYPL": ["PayPal"],
    "DIS": ["Disney", "The Walt Disney Company"],
    "NFLX": ["Netflix"],
    "NIO": ["NIO Inc."],
    "NRG": ["NRG Energy"],
    "ADBE": ["Adobe", "Adobe Inc."],
    "INTC": ["Intel", "Intel Corporation"],
    "CSCO": ["Cisco", "Cisco Systems"],
    "PFE": ["Pfizer"],
    "KO": ["Coca-Cola", "The Coca-Cola Company"],
    "PEP": ["Pepsi", "PepsiCo"],
    "MRK": ["Merck"],
    "ABT": ["Abbott", "Abbott Laboratories"],
    "XOM": ["ExxonMobil", "Exxon"],
    "CVX": ["Chevron"],
    "T": ["AT&T"],
    "MCD": ["McDonald's"],
    "NKE": ["Nike"],
    "HD": ["Home Depot"],
    "IBM": ["IBM", "International Business Machines"],
    "CRM": ["Salesforce"],
    "BMY": ["Bristol-Myers", "Bristol-Myers Squibb"],
    "ORCL": ["Oracle"],
    "ACN": ["Accenture"],
    "LLY": ["Eli Lilly"],
    "QCOM": ["Qualcomm"],
    "HON": ["Honeywell"],
    "COST": ["Costco"],
    "SBUX": ["Starbucks"],
    "CAT": ["Caterpillar"],
    "LOW": ["Lowe's"],
    "MS": ["Morgan Stanley", "Morgan Stanley Bank"],
    "GS": ["Goldman Sachs"],
    "AXP": ["American Express"],
    "INTU": ["Intuit"],
    "AMGN": ["Amgen"],
    "GE": ["General Electric"],
    "FIS": ["Fidelity National Information Services"],
    "CVS": ["CVS Health"],
    "DE": ["Deere", "John Deere"],
    "BDX": ["Becton Dickinson"],
    "NOW": ["ServiceNow"],
    "SCHW": ["Charles Schwab"],
    "LMT": ["Lockheed Martin"],
    "ADP": ["ADP", "Automatic Data Processing"],
    "C": ["Citigroup"],
    "PLD": ["Prologis"],
    "NSC": ["Norfolk Southern"],
    "TMUS": ["T-Mobile"],
    "ITW": ["Illinois Tool Works"],
    "FDX": ["FedEx"],
    "PNC": ["PNC Financial"],
    "SO": ["Southern Company"],
    "APD": ["Air Products & Chemicals"],
    "ADI": ["Analog Devices"],
    "ICE": ["Intercontinental Exchange"],
    "ZTS": ["Zoetis"],
    "TJX": ["TJX Companies"],
    "CL": ["Colgate-Palmolive"],
    "MMC": ["Marsh & McLennan"],
    "EL": ["Estée Lauder"],
    "GM": ["General Motors"],
    "CME": ["CME Group"],
    "EW": ["Edwards Lifesciences"],
    "AON": ["Aon plc"],
    "D": ["Dominion Energy"],
    "PSA": ["Public Storage"],
    "AEP": ["American Electric Power"],
    "TROW": ["T. Rowe Price"],
    "LNTH": ["Lantheus"],
    "HE": ["Hawaiian Electric"],
    "BTDR": ["Bitdeer"],
    "NAAS": ["NaaS Technology"],
    "SCHL": ["Scholastic"],
    "TGT": ["Target"],
    "SYK": ["Stryker"],
    "BKNG": ["Booking Holdings", "Booking.com"],
    "DUK": ["Duke Energy"],
    "USB": ["U.S. Bancorp"],
    "BABA": ["Alibaba", "Alibaba Group"],
    "HTZ": ["Hertz"],
    "UBER": ["Uber"],
    "LYFT": ["Lyft"],
    "PLTR": ["Palantir"],
    "SNOW": ["Snowflake"],
    "ROKU": ["Roku"],
    "TWLO": ["Twilio"],
    "SQ": ["Block", "Square"],
    "COIN": ["Coinbase", "Coinbase Global"],
    "PST.MI": ["Poste Italiane", "Poste Italiane S.p.A."],
    "UCG.MI": ["Unicredit", "UniCredit"],
    "ISP.MI": ["Intesa Sanpaolo", "Banca Intesa"],
    "ENEL.MI": ["Enel"],
    "STLAM.MI": ["Stellantis", "Fiat Chrysler"],
    "LDO.MI": ["Leonardo", "Leonardo Finmeccanica"],
    "RIVN": ["Rivian"],
    "LCID": ["Lucid", "Lucid Motors"],
    "DDOG": ["Datadog"],
    "NET": ["Cloudflare"],
    "SHOP": ["Shopify"],
    "ZI": ["ZoomInfo"],
    "ZM": ["Zoom Video"],
    "BIDU": ["Baidu"],
    "PDD": ["Pinduoduo", "PDD Holdings"],
    "JD": ["JD.com"],
    "ARM": ["Arm", "Arm Holdings"],
    "DUOL": ["Duolingo"],
    "PBR": ["Petrobras"],
    "VALE": ["Vale S.A."],
    "AMX": ["America Movil"],

    # Forex
    "EURUSD": ["EUR/USD", "Euro Dollar"],
    "USDJPY": ["USD/JPY", "Dollar Yen"],
    "GBPUSD": ["GBP/USD", "British Pound"],
    "AUDUSD": ["AUD/USD", "Australian Dollar"],
    "USDCAD": ["USD/CAD", "Canadian Dollar"],
    "USDCHF": ["USD/CHF", "Swiss Franc"],
    "NZDUSD": ["NZD/USD", "New Zealand Dollar"],
    "EURGBP": ["EUR/GBP"],
    "EURJPY": ["EUR/JPY"],

    #Index
    "SPX500": ["S&P 500", "SPX", "US Market"],
    "DJ30": ["Dow Jones", "DJIA", "Dow 30"],
    "NAS100": ["Nasdaq 100", "NDX"],
    "NASCOMP": ["Nasdaq Composite", "IXIC"],
    "RUS2000": ["Russell 2000", "RUT"],
    "VIX": ["VIX", "Volatility Index"],
    "EU50": ["Euro Stoxx 50"],
    "ITA40": ["FTSE MIB", "Milan Index"],
    "GER40": ["DAX 40", "German DAX"],
    "UK100": ["FTSE 100"],
    "FRA40": ["CAC 40"],
    "SWI20": ["Swiss Market Index", "SMI"],
    "ESP35": ["IBEX 35"],
    "NETH25": ["AEX"],
    "JPN225": ["Nikkei 225"],
    "HKG50": ["Hang Seng"],
    "CHN50": ["Shanghai Composite"],
    "IND50": ["Nifty 50"],
    "KOR200": ["KOSPI"],
    
    # Crypto
    "BTCUSD": ["Bitcoin", "BTC"],
    "ETHUSD": ["Ethereum", "ETH"],
    "LTCUSD": ["Litecoin", "LTC"],
    "XRPUSD": ["Ripple", "XRP"],
    "BCHUSD": ["Bitcoin Cash"],
    "SOLUSD": ["Solana", "SOL"],
    "DOGEUSD": ["Dogecoin", "DOGE"],
    "USDTUSD": ["Tether", "USDT"],

    # Commodities
    "COCOA": ["Cocoa Futures"],
    "GOLD": ["Gold Price", "XAU/USD"],
    "SILVER": ["Silver Price", "XAG/USD"],
    "OIL": ["Crude Oil", "WTI", "Brent"],
    "NATGAS": ["Natural Gas"]
}

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
    
    # 2. Trova chi è il leader "friendly"
    friendly_leader = sector_leaders.get(sector, "SPX500") 
    
    # 3. Converti il leader friendly in Yahoo Ticker
    yahoo_leader = TICKER_MAP.get(friendly_leader, friendly_leader)
    
    return sector, yahoo_leader

def get_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        
        # Gestione MultiIndex di yfinance
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

def fetch_rss(url):
    """Funzione helper per scaricare e parsare RSS"""
    titles = []
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=2)
        if resp.status_code != 200: return []
        root = ET.fromstring(resp.content)
        now = datetime.now().astimezone()
        for item in root.findall('.//item'):
            try:
                pub_txt = item.find('pubDate').text
                pd_date = parsedate_to_datetime(pub_txt)
                # Filtra news ultime 48h
                if (now - pd_date) < timedelta(hours=48): 
                    titles.append(item.find('title').text)
            except: continue
    except:
        pass
    return titles

def get_news_data_super_charged(ticker_yahoo, friendly_symbol, sector):
    """
    Esegue 2 ricerche diverse e unisce i risultati per massimizzare le news.
    Usa TUTTI gli identificativi disponibili: Ticker Friendly, Ticker Yahoo e Nomi Estesi.
    """
    
    # --- 1. COSTRUZIONE CANDIDATI RICERCA ---
    candidates = []
    
    # A. Simbolo User Friendly (es. "SPX500")
    candidates.append(friendly_symbol)
    
    # B. Ticker Yahoo (Pulito) - Utile per titoli italiani (es. "ISP.MI")
    # Escludiamo ticker con simboli strani se non strettamente necessari
    clean_yahoo = ticker_yahoo.replace("=X", "").replace("^", "")
    if clean_yahoo != friendly_symbol:
        candidates.append(clean_yahoo)
        
    # C. Nomi Estesi (es. "Intesa Sanpaolo")
    mapped_names = symbol_name_map.get(friendly_symbol, [])
    candidates.extend(mapped_names)
    
    # Rimuovi duplicati e stringhe vuote
    names = []
    for c in candidates:
        if c and c not in names:
            names.append(c)
            
    # Set di titoli unici per evitare duplicati nei risultati
    unique_titles = set()
    
    # --- 2. CONFIGURAZIONE KEYWORDS PER SETTORE ---
    if "Crypto" in sector:
        keywords_pass1 = "crypto"
        keywords_pass2 = "price prediction OR market OR news"
    elif "Forex" in sector:
        keywords_pass1 = "forex"
        keywords_pass2 = "exchange rate OR forecast"
    elif "Indices" in sector:
        keywords_pass1 = "market index"
        keywords_pass2 = "stock market OR analysis"
    elif "Commodities" in sector or "Energy" in sector or "Metals" in sector:
        keywords_pass1 = "price"
        keywords_pass2 = "futures OR commodities"
    else:
        # Stock
        keywords_pass1 = "stock"
        keywords_pass2 = "earnings OR shares OR analysis"

    # Preparazione termini ricerca (limitati a 3 per query per evitare errori URL troppo lunghi)
    search_terms = [f'"{n}"' for n in names[:3]]
    base_query = " OR ".join(search_terms)

    # --- 3. ESECUZIONE QUERY ---
    
    # QUERY A: Specifica (es. "Bitcoin" crypto)
    q1 = f"({base_query}) {keywords_pass1}"
    url1 = f"https://news.google.com/rss/search?q={q1}&hl=en-US&gl=US&ceid=US:en"
    
    # QUERY B: Ampia (es. "Bitcoin" news OR price)
    q2 = f"({base_query}) ({keywords_pass2})"
    url2 = f"https://news.google.com/rss/search?q={q2}&hl=en-US&gl=US&ceid=US:en"
    
    # Esecuzione
    titles1 = fetch_rss(url1)
    titles2 = fetch_rss(url2)
    
    # Unione risultati
    for t in titles1: unique_titles.add(t)
    for t in titles2: unique_titles.add(t)
    
    final_titles = list(unique_titles)
    count = len(final_titles)
    
    if count == 0: return 0.0, 0
    
    # Analisi Sentiment
    sia = SentimentIntensityAnalyzer()
    lexicon = {
        'surge': 4.0, 'jump': 2.0, 'rally': 3.5, 'soar': 4.0, 'bull': 3.0, 'buy': 2.0,
        'plunge': -4.0, 'crash': -4.0, 'drop': -3.0, 'bear': -3.0, 'sell': -2.0,
        'miss': -2.0, 'beat': 2.0, 'strong': 1.5, 'weak': -1.5, 'record': 2.0,
        'high': 1.0, 'low': -1.0, 'gain': 2.0, 'loss': -2.0, 'up': 1.0, 'down': -1.0
    }
    sia.lexicon.update(lexicon)
    
    total = sum([sia.polarity_scores(t)['compound'] for t in final_titles])
    return (total / count), count

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
        
        # Logica Pesi: Più news ho, più il sentiment pesa. Se ho 0 news, mi fido del tecnico/leader.
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
    print(f"\n--- ANALISI PORTAFOGLIO SETTORIALE ({datetime.now().strftime('%Y-%m-%d')}) ---")
    
    results = []
    leader_cache = {}
    WORK_LIST = []
    
    for friendly_name in USER_SYMBOL_LIST:
        yahoo_ticker = TICKER_MAP.get(friendly_name, friendly_name)
        sec_name, leader_yf_tick = get_sector_and_leader(friendly_name)
        WORK_LIST.append({
            "friendly": friendly_name, "yahoo": yahoo_ticker,
            "sec": sec_name, "bench": leader_yf_tick
        })

    WORK_LIST.sort(key=lambda x: x['sec'])
    current_sector = ""
    
    for item in WORK_LIST:
        friendly = item['friendly']
        yahoo_t = item['yahoo']
        sec = item['sec']
        bench = item['bench']

        if sec != current_sector:
            if bench not in leader_cache:
                leader_cache[bench] = get_leader_trend(bench)
            ld_score = leader_cache[bench]
            icon = "🟢 UP" if ld_score > 0 else "🔴 DOWN"
            print(f"\n📂 {sec}")
            print(f"   👑 LEADER TREND ({bench}): {icon}")
            print("-" * 85)
            current_sector = sec
        else:
            ld_score = leader_cache[bench]

        df = get_data(yahoo_t)
        
        if not df.empty:
            # USIAMO LA NUOVA FUNZIONE SUPER CHARGED
            sentiment, count = get_news_data_super_charged(yahoo_t, friendly, sec)
            
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
            
            lead_mark = "👑" if is_leader else ""
            print(f"   {friendly:<10} {lead_mark:<2} | {prob}% | {sig:<11} | News:{count:<3} | Sent:{sent}")
        else:
            print(f"   {friendly:<10}    | ⚠️ DATA ERROR ({yahoo_t})")
        
        # Leggera pausa per evitare blocco IP Google
        time.sleep(0.1) 

    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by="Score", ascending=False)
        print("\n\n" + "="*100)
        print(f"🏆 TOP OPPORTUNITIES (GLOBAL RANKING)")
        print("="*100)
        print(f"{'ASSET':<12} | {'SECTOR':<25} | {'SCORE':<6} | {'SIGNAL':<11} | {'NEWS':<4} | {'SENT'}")
        print("-" * 100)
        for _, row in df_res.iterrows():
            icon = "🟢" if "BUY" in row['Signal'] else "🔴" if "SELL" in row['Signal'] else "⚪"
            print(f"{row['Asset']:<12} | {row['Sector'][:25]:<25} | {row['Score']:<6} | {icon} {row['Signal']:<9} | {row['News']:<4} | {row['Sent']}")
