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
# 1. IL CUORE DEL SISTEMA: LA MAPPA DI CONVERSIONE (TICKER_MAP)
# ==============================================================================
TICKER_MAP = {
    # --- US Stocks (Tech & General) ---
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
    "CHTR": "CHTR", # Aggiunto per completezza telecom

    # --- Growth / New Tech ---
    "ARM": "ARM", "BABA": "BABA", "BIDU": "BIDU", "COIN": "COIN", "DDOG": "DDOG", 
    "HTZ": "HTZ", "JD": "JD", "LCID": "LCID", "LYFT": "LYFT", "NET": "NET", 
    "PDD": "PDD", "PLTR": "PLTR", "RIVN": "RIVN", "ROKU": "ROKU", "SHOP": "SHOP", 
    "SNOW": "SNOW", "SQ": "SQ", "TWLO": "TWLO", "UBER": "UBER", "ZI": "ZI", 
    "ZM": "ZM", "DUOL": "DUOL", "PBR": "PBR", "VALE": "VALE", "AMX": "AMX",

    # --- Europe / Italy ---
    "ISP.MI": "ISP.MI", "ENEL.MI": "ENEL.MI", "STLAM.MI": "STLAM.MI", 
    "LDO.MI": "LDO.MI", "PST.MI": "PST.MI", "UCG.MI": "UCG.MI",

    # --- Forex (Markets) ---
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

    # --- Commodities / Leaders ---
    # Nota: OIL è il Leader dell'Energy, GOLD dei metalli.
    "COCOA": "CC=F", "GOLD": "GC=F", "SILVER": "SI=F", "OIL": "CL=F", "NATGAS": "NG=F"
}

# ==============================================================================
# 2. CONFIGURAZIONE SETTORI E LEADER
# ==============================================================================

# Definisce chi comanda ogni settore (Usiamo il nome "friendly" che poi viene mappato)
sector_leaders = {
    "1. Big Tech, Software & Internet": "MSFT",  # Microsoft è il re del software
    "2. Semiconductors & AI": "NVDA",           # Nvidia guida i chip
    "3. Financial Services": "JPM",             # JP Morgan guida la finanza
    "4. Automotive & Mobility": "TSLA",         # Tesla guida il sentiment auto
    "5. Healthcare & Pharma": "LLY",            # Eli Lilly (obesità/pharma growth)
    "6. Consumer Goods & Retail": "WMT",        # Walmart è l'economia reale
    "7. Industrials & Defense": "CAT",          # Caterpillar è il termometro industriale
    "8. Energy (Oil & Gas)": "OIL",             # Il prezzo del petrolio comanda qui
    "9. Utilities & Green": "SO",               # Southern Co per le utilities classiche
    "10. Precious Metals & Materials": "GOLD",  # L'oro guida i metalli
    "11. Media & Telecom": "NFLX",              # Netflix guida l'intrattenimento
    "12. Indices (Global)": "SPX500",           # S&P500 è il mercato
    "13. Forex (Currencies)": "EURUSD",         # L'Euro/Dollaro è il re del FX
    "14. Crypto Assets": "BTCUSD"               # Bitcoin guida le crypto
}

# Mappa ogni singolo asset al suo nuovo settore
asset_sector_map = {
    # --- 1. Big Tech, Software & Internet (Leader: MSFT) ---
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

    # --- 2. Semiconductors & AI (Leader: NVDA) ---
    "NVDA": "2. Semiconductors & AI", "INTC": "2. Semiconductors & AI",
    "QCOM": "2. Semiconductors & AI", "ADI": "2. Semiconductors & AI",
    "ARM": "2. Semiconductors & AI", "CSCO": "2. Semiconductors & AI",
    "ACN": "2. Semiconductors & AI", "FIS": "2. Semiconductors & AI",

    # --- 3. Financial Services (Leader: JPM) ---
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

    # --- 4. Automotive & Mobility (Leader: TSLA) ---
    "TSLA": "4. Automotive & Mobility", "GM": "4. Automotive & Mobility",
    "NIO": "4. Automotive & Mobility", "STLAM.MI": "4. Automotive & Mobility",
    "HTZ": "4. Automotive & Mobility", "LCID": "4. Automotive & Mobility",
    "RIVN": "4. Automotive & Mobility", "UBER": "4. Automotive & Mobility",
    "LYFT": "4. Automotive & Mobility", "NAAS": "4. Automotive & Mobility",

    # --- 5. Healthcare & Pharma (Leader: LLY) ---
    "LLY": "5. Healthcare & Pharma", "JNJ": "5. Healthcare & Pharma",
    "PFE": "5. Healthcare & Pharma", "MRK": "5. Healthcare & Pharma",
    "ABT": "5. Healthcare & Pharma", "BMY": "5. Healthcare & Pharma",
    "AMGN": "5. Healthcare & Pharma", "CVS": "5. Healthcare & Pharma",
    "BDX": "5. Healthcare & Pharma", "ZTS": "5. Healthcare & Pharma",
    "EW": "5. Healthcare & Pharma", "LNTH": "5. Healthcare & Pharma",
    "SYK": "5. Healthcare & Pharma",

    # --- 6. Consumer Goods & Retail (Leader: WMT) ---
    "WMT": "6. Consumer Goods & Retail", "KO": "6. Consumer Goods & Retail",
    "PEP": "6. Consumer Goods & Retail", "MCD": "6. Consumer Goods & Retail",
    "NKE": "6. Consumer Goods & Retail", "HD": "6. Consumer Goods & Retail",
    "COST": "6. Consumer Goods & Retail", "SBUX": "6. Consumer Goods & Retail",
    "LOW": "6. Consumer Goods & Retail", "TGT": "6. Consumer Goods & Retail",
    "TJX": "6. Consumer Goods & Retail", "CL": "6. Consumer Goods & Retail",
    "EL": "6. Consumer Goods & Retail", "SCHL": "6. Consumer Goods & Retail",
    "COCOA": "6. Consumer Goods & Retail", # Spostato qui come richiesto

    # --- 7. Industrials & Defense (Leader: CAT) ---
    "CAT": "7. Industrials & Defense", "LMT": "7. Industrials & Defense",
    "ITW": "7. Industrials & Defense", "FDX": "7. Industrials & Defense",
    "NSC": "7. Industrials & Defense", "GE": "7. Industrials & Defense",
    "HON": "7. Industrials & Defense", "DE": "7. Industrials & Defense",
    "LDO.MI": "7. Industrials & Defense", "BKNG": "7. Industrials & Defense",

    # --- 8. Energy (Oil & Gas) (Leader: OIL) ---
    "OIL": "8. Energy (Oil & Gas)", "NATGAS": "8. Energy (Oil & Gas)",
    "XOM": "8. Energy (Oil & Gas)", "CVX": "8. Energy (Oil & Gas)",
    "PBR": "8. Energy (Oil & Gas)", "NRG": "8. Energy (Oil & Gas)",

    # --- 9. Utilities & Green (Leader: SO) ---
    "SO": "9. Utilities & Green", "ENEL.MI": "9. Utilities & Green",
    "DUK": "9. Utilities & Green", "AEP": "9. Utilities & Green",
    "D": "9. Utilities & Green", "HE": "9. Utilities & Green",
    "APD": "9. Utilities & Green",

    # --- 10. Precious Metals & Materials (Leader: GOLD) ---
    "GOLD": "10. Precious Metals & Materials", "SILVER": "10. Precious Metals & Materials",
    "VALE": "10. Precious Metals & Materials",

    # --- 11. Media & Telecom (Leader: NFLX) ---
    "NFLX": "11. Media & Telecom", "DIS": "11. Media & Telecom",
    "T": "11. Media & Telecom", "TMUS": "11. Media & Telecom",
    "AMX": "11. Media & Telecom", "ROKU": "11. Media & Telecom",

    # --- 12. Indices (Global) (Leader: SPX500) ---
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

    # --- 13. Forex (Currencies) (Leader: EURUSD) ---
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

    # --- 14. Crypto Assets (Leader: BTCUSD) ---
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

# Per semplicità, la search map per le news usa il nome friendly
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
    "MS": ["Morgan Stanley", "Morgan Stanley Bank", "MS bank", "MS financial"],
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
    "BABA": ["Alibaba", "Alibaba Group", "阿里巴巴"],
    "HTZ": ["Hertz", "Hertz Global", "Hertz Global Holdings"],
    "UBER": ["Uber", "Uber Technologies", "Uber Technologies Inc."],
    "LYFT": ["Lyft", "Lyft Inc."],
    "PLTR": ["Palantir", "Palantir Technologies", "Palantir Technologies Inc."],
    "SNOW": ["Snowflake", "Snowflake Inc."],
    "ROKU": ["Roku", "Roku Inc."],
    "TWLO": ["Twilio", "Twilio Inc."],
    "SQ": ["Block", "Square", "Block Inc.", "Square Inc."],
    "COIN": ["Coinbase", "Coinbase Global", "Coinbase Global Inc."],
    "PST.MI": ["Poste Italiane", "Poste Italiane S.p.A."],
    "UCG.MI": ["Unicredit", "UniCredit", "Unicredit S.p.A.", "UniCredit Bank"],
    "ISP.MI": ["Intesa Sanpaolo", "Intesa Sanpaolo S.p.A.", "Gruppo Intesa Sanpaolo", "Intesa Sanpaolo Bank", "Banca Intesa", "Banca Sanpaolo"],
    "ENEL.MI": ["Enel", "Enel S.p.A.", "Gruppo Enel"],
    "STLAM.MI": ["Stellantis", "Stellantis N.V.", "Gruppo Stellantis", "Fiat Chrysler", "FCA", "PSA Group"],
    "LDO.MI": ["Leonardo", "Leonardo S.p.A.", "Leonardo Finmeccanica", "Gruppo Leonardo"],
    "RIVN": ["Rivian", "Rivian Automotive", "Rivian Automotive Inc."],
    "LCID": ["Lucid", "Lucid Motors", "Lucid Group", "Lucid Group Inc."],
    "DDOG": ["Datadog", "Datadog Inc."],
    "NET": ["Cloudflare", "Cloudflare Inc."],
    "SHOP": ["Shopify", "Shopify Inc."],
    "ZI": ["ZoomInfo", "ZoomInfo Technologies", "ZoomInfo Technologies Inc."],
    "ZM": ["Zoom", "Zoom Video", "Zoom Video Communications", "Zoom Video Communications Inc."],
    "BIDU": ["Baidu", "百度"],
    "PDD": ["Pinduoduo", "PDD Holdings", "Pinduoduo Inc.", "拼多多"],
    "JD": ["JD.com", "京东"],
    "ARM": ["Arm", "Arm Holdings", "Arm Holdings plc"],
    "DUOL": ["Duolingo", "Duolingo Inc.", "DUOL"],
    "PBR": ["Petrobras", "Petróleo Brasileiro S.A.", "Petrobras S.A."],
    "VALE": ["Vale", "Vale S.A.", "Vale SA"],
    "AMX": ["America Movil", "América Móvil", "América Móvil S.A.B. de C.V."],

    # Forex
    "EURUSD": ["EUR/USD", "Euro Dollar", "Euro vs USD"],
    "USDJPY": ["USD/JPY", "Dollar Yen", "USD vs JPY"],
    "GBPUSD": ["GBP/USD", "British Pound", "Sterling", "GBP vs USD"],
    "AUDUSD": ["AUD/USD", "Australian Dollar", "Aussie Dollar"],
    "USDCAD": ["USD/CAD", "US Dollar vs Canadian Dollar", "Loonie"],
    "USDCHF": ["USD/CHF", "US Dollar vs Swiss Franc"],
    "NZDUSD": ["NZD/USD", "New Zealand Dollar"],
    "EURGBP": ["EUR/GBP", "Euro vs Pound"],
    "EURJPY": ["EUR/JPY", "Euro vs Yen"],
    "GBPJPY": ["GBP/JPY", "Pound vs Yen"],
    "AUDJPY": ["AUD/JPY", "Aussie vs Yen"],
    "CADJPY": ["CAD/JPY", "Canadian Dollar vs Yen"],
    "CHFJPY": ["CHF/JPY", "Swiss Franc vs Yen"],
    "EURAUD": ["EUR/AUD", "Euro vs Aussie"],
    "EURNZD": ["EUR/NZD", "Euro vs Kiwi"],
    "EURCAD": ["EUR/CAD", "Euro vs Canadian Dollar"],
    "EURCHF": ["EUR/CHF", "Euro vs Swiss Franc"],
    "GBPCHF": ["GBP/CHF", "Pound vs Swiss Franc"],
    "AUDCAD": ["AUD/CAD", "Aussie vs Canadian Dollar"],

    #Index
    "SPX500": ["S&P 500", "SPX", "S&P", "S&P 500 Index", "Standard & Poor's 500"],
    "DJ30": ["Dow Jones", "DJIA", "Dow Jones Industrial", "Dow 30", "Dow Jones Industrial Average"],
    "NAS100": ["Nasdaq 100", "NDX", "Nasdaq100", "NASDAQ 100 Index"],
    "NASCOMP": ["Nasdaq Composite", "IXIC", "Nasdaq", "Nasdaq Composite Index"],
    "RUS2000": ["Russell 2000", "RUT", "Russell Small Cap", "Russell 2K"],
    "VIX": ["VIX", "Volatility Index", "Fear Gauge", "CBOE Volatility Index"],
    "EU50": ["Euro Stoxx 50", "Euro Stoxx", "STOXX50", "Euro Stoxx 50 Index"],
    "ITA40": ["FTSE MIB", "MIB", "FTSE MIB Index", "Italy 40"],
    "GER40": ["DAX", "DAX 40", "German DAX", "Frankfurt DAX"],
    "UK100": ["FTSE 100", "FTSE", "UK FTSE 100", "FTSE Index"],
    "FRA40": ["CAC 40", "CAC", "France CAC 40", "CAC40 Index"],
    "SWI20": ["Swiss Market Index", "SMI", "Swiss SMI", "Swiss Market"],
    "ESP35": ["IBEX 35", "IBEX", "Spanish IBEX", "IBEX 35 Index"],
    "NETH25": ["AEX", "Dutch AEX", "Amsterdam Exchange", "AEX Index"],
    "JPN225": ["Nikkei 225", "Nikkei", "Japan Nikkei", "Nikkei Index"],
    "HKG50": ["Hang Seng", "Hong Kong Hang Seng", "Hang Seng Index"],
    "CHN50": ["Shanghai Composite", "SSEC", "China Shanghai", "Shanghai Composite Index"],
    "IND50": ["Nifty 50", "Nifty", "India Nifty", "Nifty 50 Index"],
    "KOR200": ["KOSPI", "KOSPI 200", "Korea KOSPI", "KOSPI Index"],
    
    # Crypto
    "BTCUSD": ["Bitcoin", "BTC"],
    "ETHUSD": ["Ethereum", "ETH"],
    "LTCUSD": ["Litecoin", "LTC"],
    "XRPUSD": ["Ripple", "XRP"],
    "BCHUSD": ["Bitcoin Cash", "BCH"],
    "EOSUSD": ["EOS"],
    "XLMUSD": ["Stellar", "XLM"],
    "ADAUSD": ["Cardano", "ADA"],
    "TRXUSD": ["Tron", "TRX"],
    "NEOUSD": ["NEO"],
    "DASHUSD": ["Dash crypto", "Dash cryptocurrency", "DASH coin", "DASH token", "Digital Cash", "Dash blockchain", "Dash digital currency"],
    "XMRUSD": ["Monero", "XMR"],
    "ETCUSD": ["Ethereum Classic", "ETC"],
    "ZECUSD": ["Zcash", "ZEC"],
    "BNBUSD": ["Binance Coin", "BNB"],
    "DOGEUSD": ["Dogecoin", "DOGE"],
    "USDTUSD": ["Tether", "USDT"],
    "LINKUSD": ["Chainlink", "LINK"],
    "ATOMUSD": ["Cosmos", "ATOM"],
    "XTZUSD": ["Tezos", "XTZ"],

    # Commodities
    "COCOA": ["Cocoa", "Cocoa Futures"],
    "GOLD": ["Gold", "XAU/USD", "Gold price", "Gold spot"],
    "SILVER": ["Silver", "XAG/USD", "Silver price", "Silver spot"],
    "OIL": ["Crude oil", "Oil price", "WTI", "Brent", "Brent oil", "WTI crude"],
    "NATGAS": ["Natural gas", "Gas price", "Natgas", "Henry Hub", "NG=F", "Natural gas futures"]
}

USER_SYMBOL_LIST = list(asset_sector_map.keys())

# ==============================================================================
# 3. UTILITY FUNZIONI (Logica invariata)
# ==============================================================================

def get_sector_and_leader(friendly_symbol):
    """
    Trova il settore e il ticker Yahoo del leader di quel settore.
    """
    # 1. Trova il settore
    sector = asset_sector_map.get(friendly_symbol, "Unknown Sector")
    
    # 2. Trova chi è il leader "friendly"
    friendly_leader = sector_leaders.get(sector, "SPX500") 
    
    # 3. Converti il leader friendly in Yahoo Ticker (es. 'OIL' -> 'CL=F')
    yahoo_leader = TICKER_MAP.get(friendly_leader, friendly_leader)
    
    return sector, yahoo_leader

def get_data(ticker):
    try:
        # Nota: auto_adjust=True corregge split e dividendi
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        
        # Gestione MultiIndex di yfinance recente
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
        # Logica Trend: +0.5 se sopra media 50 periodi, -0.5 se sotto
        return 0.5 if curr > sma else -0.5
    except: return 0.0

def get_news_data_advanced(ticker_yahoo, friendly_symbol):
    # Genera una query semplice usando il simbolo amichevole
    search_query = f'"{friendly_symbol}" stock'
    rss_url = f"https://news.google.com/rss/search?q=({search_query})&hl=en-US&gl=US&ceid=US:en"
    
    try:
        resp = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=2)
        root = ET.fromstring(resp.content)
        titles = []
        now = datetime.now().astimezone()
        
        for item in root.findall('.//item'):
            try:
                pub_txt = item.find('pubDate').text
                pd_date = parsedate_to_datetime(pub_txt)
                # Filtra news ultime 48h
                if (now - pd_date) < timedelta(hours=48): 
                    titles.append(item.find('title').text)
            except: continue
        
        count = len(titles)
        if count == 0: return 0.0, 0
        
        sia = SentimentIntensityAnalyzer()
        # Dizionario finanziario potenziato
        lexicon = {
            'surge': 4.0, 'jump': 2.0, 'rally': 3.5, 'soar': 4.0, 'bull': 3.0, 'buy': 2.0,
            'plunge': -4.0, 'crash': -4.0, 'drop': -3.0, 'bear': -3.0, 'sell': -2.0,
            'miss': -2.0, 'beat': 2.0, 'strong': 1.5, 'weak': -1.5
        }
        sia.lexicon.update(lexicon)
        
        total = sum([sia.polarity_scores(t)['compound'] for t in titles])
        return (total / count), count
        
    except Exception as e:
        return 0.0, 0

# ==============================================================================
# 4. ENGINE DI CALCOLO (Logica invariata)
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
        # 1. Trend tecnico (SMA)
        if curr > sma: score += 0.5
        else: score -= 0.5
        # 2. Ipercomprato/Ipervenduto (RSI)
        if rsi < 30: score += 0.5 # Buy dip
        elif rsi > 70: score -= 0.5 # Sell top
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df, sent, news_n, lead, is_lead):
        tech = self._get_technical_score(df)
        
        # Se sono il leader, il mio trend è me stesso (lead è già calcolato su di me)
        # Se NON sono il leader, uso il trend del leader (lead)
        curr_lead = 0.0 if is_lead else lead
        
        # Pesi dinamici
        if is_lead:
            # Il leader si fida più di se stesso (tecnica) e news
            if news_n == 0: w_n, w_l, w_t = 0.0, 0.0, 1.0
            elif news_n <= 3: w_n, w_l, w_t = 0.30, 0.0, 0.70
            else: w_n, w_l, w_t = 0.60, 0.0, 0.40
        else:
            # I follower dipendono dal leader
            if news_n == 0: w_n, w_l, w_t = 0.0, 0.35, 0.65
            elif news_n <= 3: w_n, w_l, w_t = 0.20, 0.25, 0.55
            else: w_n, w_l, w_t = 0.55, 0.15, 0.30
        
        final = (sent * w_n) + (tech * w_t) + (curr_lead * w_l)
        final = max(min(final, 1.0), -1.0)
        # Normalizza su scala 0-100%
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
    
    # Prepara la lista di lavoro
    for friendly_name in USER_SYMBOL_LIST:
        yahoo_ticker = TICKER_MAP.get(friendly_name, friendly_name)
        sec_name, leader_yf_tick = get_sector_and_leader(friendly_name)
        
        WORK_LIST.append({
            "friendly": friendly_name,
            "yahoo": yahoo_ticker,
            "sec": sec_name,
            "bench": leader_yf_tick
        })

    # Ordina per settore
    WORK_LIST.sort(key=lambda x: x['sec'])
    
    current_sector = ""
    
    for item in WORK_LIST:
        friendly = item['friendly']
        yahoo_t = item['yahoo']
        sec = item['sec']
        bench = item['bench']

        # Stampa intestazione settore e trend leader
        if sec != current_sector:
            # Recupera trend del leader se non in cache
            if bench not in leader_cache:
                leader_cache[bench] = get_leader_trend(bench)
            
            ld_score = leader_cache[bench]
            icon = "🟢 UP" if ld_score > 0 else "🔴 DOWN"
            print(f"\n📂 {sec}")
            print(f"   👑 LEADER TREND ({bench}): {icon}")
            print("-" * 80)
            current_sector = sec
        else:
            ld_score = leader_cache[bench]

        # Scarica dati
        df = get_data(yahoo_t)
        
        if not df.empty:
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
            
            # Formattazione condizionale output
            lead_mark = "👑" if is_leader else ""
            print(f"   {friendly:<10} {lead_mark:<2} | {prob}% | {sig:<11} | News:{count}")
        else:
            print(f"   {friendly:<10}    | ⚠️ DATA ERROR ({yahoo_t})")
        
        time.sleep(0.05)

    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by="Score", ascending=False)
        print("\n\n" + "="*100)
        print(f"🏆 TOP OPPORTUNITIES (GLOBAL RANKING)")
        print("="*100)
        print(f"{'ASSET':<12} | {'SECTOR':<25} | {'SCORE':<6} | {'SIGNAL':<11} | {'NEWS':<4}")
        print("-" * 100)
        for _, row in df_res.iterrows():
            icon = "🟢" if "BUY" in row['Signal'] else "🔴" if "SELL" in row['Signal'] else "⚪"
            print(f"{row['Asset']:<12} | {row['Sector'][:25]:<25} | {row['Score']:<6} | {icon} {row['Signal']:<9} | {row['News']:<4}")
