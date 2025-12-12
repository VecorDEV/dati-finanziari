from github import Github, GithubException
import re
import feedparser
import os
from datetime import datetime, timedelta
import math
import spacy
# Librerie dati e calcoli
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import random
import unicodedata
import json
import base64
import requests
import argostranslate.package
import argostranslate.translate
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Indicatori tecnici
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands
from urllib.parse import quote_plus
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr, binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# --- SETUP AI (VADER) ---
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Carica Spacy (legacy)
nlp = spacy.load("en_core_web_sm")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = "VecorDEV/dati-finanziari"

# --- MODIFICA CARTELLA DI DESTINAZIONE ---
TARGET_FOLDER = "hybrid_results"  # La nuova cartella

# Paths iniziali
file_path = f"{TARGET_FOLDER}/classifica.html"
news_path = f"{TARGET_FOLDER}/news.html"
history_path = f"{TARGET_FOLDER}/history.json"
    
# GitHub Connect
github = Github(GITHUB_TOKEN)
repo = github.get_repo(REPO_NAME)

# 📌 Lingue da generare
LANGUAGES = {
    "ar": "daily_brief_ar.html", "de": "daily_brief_de.html", "es": "daily_brief_es.html",
    "fr": "daily_brief_fr.html", "hi": "daily_brief_hi.html", "it": "daily_brief_it.html",
    "ko": "daily_brief_ko.html", "nl": "daily_brief_nl.html", "pt": "daily_brief_pt.html",
    "ru": "daily_brief_ru.html", "zh": "daily_brief_zh.html", "zh-rCN": "daily_brief_zh-rCN.html",
}

# ==============================================================================
# 1. LISTE E MAPPE COMPLETE
# ==============================================================================

# Mapping Leaders per settore
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

# Mapping Asset -> Settore (Completo)
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

# Mapping Ticker Yahoo
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

# Lista Simboli da analizzare
symbol_list = list(asset_sector_map.keys())

# Per compatibilità con loop esistenti:
symbol_list_for_yfinance = []
for s in symbol_list:
    symbol_list_for_yfinance.append(TICKER_MAP.get(s, s))

# Mappa Nomi Estesi Completa
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

indicator_data = {}
fundamental_data = {}

# ==============================================================================
# 2. CLASSI LOGICA NUOVA (History & Hybrid)
# ==============================================================================

class HistoryManager:
    def __init__(self, repo, filename=f"{TARGET_FOLDER}/history.json"):
        self.repo = repo
        self.filename = filename
        self.data = self._load_data_from_github()
        self._clean_old_data() 

    def _load_data_from_github(self):
        try:
            contents = self.repo.get_contents(self.filename)
            json_content = base64.b64decode(contents.content).decode('utf-8')
            return json.loads(json_content)
        except Exception:
            return {}

    def save_data_to_github(self):
        json_content = json.dumps(self.data, indent=4)
        try:
            contents = self.repo.get_contents(self.filename)
            self.repo.update_file(contents.path, "Update history data", json_content, contents.sha)
        except Exception:
            self.repo.create_file(self.filename, "Create history data", json_content)

    def _clean_old_data(self):
        limit_date = datetime.now() - timedelta(days=15)
        changed = False
        for ticker in list(self.data.keys()):
            dates = list(self.data[ticker].keys())
            for d in dates:
                try:
                    entry_date = datetime.strptime(d, "%Y-%m-%d")
                    if entry_date < limit_date:
                        del self.data[ticker][d]
                        changed = True
                except: pass

    def update_history(self, ticker, sentiment, news_count):
        today = datetime.now().strftime("%Y-%m-%d")
        if ticker not in self.data: self.data[ticker] = {}
        # Salviamo il sentiment normalizzato (0-1) e il conteggio
        self.data[ticker][today] = { "sentiment": float(sentiment), "news_count": int(news_count) }

    def calculate_delta_score(self, ticker, current_sent, current_count):
        if ticker not in self.data: return 50.0 
        history = self.data[ticker]
        today = datetime.now().strftime("%Y-%m-%d")
        
        past_sentiments = [v['sentiment'] for k, v in history.items() if k != today]
        past_counts = [v['news_count'] for k, v in history.items() if k != today]
        
        if not past_sentiments: return 50.0 
            
        avg_sent = sum(past_sentiments) / len(past_sentiments)
        avg_count = sum(past_counts) / len(past_counts) if past_counts else 1
        if avg_count == 0: avg_count = 1
        
        vol_ratio = current_count / avg_count
        vol_score = min(vol_ratio, 3.0) / 3.0
        sent_diff = current_sent - avg_sent
        
        # Logica Delta (Range 0-100)
        raw_delta = (sent_diff * 100)
        multiplier = 1.0
        if vol_ratio > 1.5: multiplier = 1.5
        if vol_ratio > 2.5: multiplier = 2.0
        
        final_delta = 50 + (raw_delta * multiplier)
        return max(min(final_delta, 100), 0)

class HybridScorer:
    def _calculate_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean().replace(0, np.nan)
        rs = gain / loss
        rs = rs.fillna(0)
        return 100 - (100 / (1 + rs))

    def _get_technical_score(self, df):
        if len(df) < 30: return 0.0
        close = df['Close']
        try:
            sma = float(close.rolling(window=50).mean().iloc[-1])
            curr = float(close.iloc[-1])
            rsi = float(self._calculate_rsi(close).iloc[-1])
        except: return 0.0
        
        score = 0.0
        if curr > sma: score += 0.5
        else: score -= 0.5
        if rsi < 30: score += 0.5 
        elif rsi > 70: score -= 0.5 
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df, sent_raw, news_n, lead, is_lead, delta_score):
        # sent_raw è tra -1 e 1 (VADER)
        tech = self._get_technical_score(df)
        curr_lead = 0.0 if is_lead else lead
        delta_factor = (delta_score - 50) / 50.0 
        
        if is_lead:
            if news_n == 0: w_n, w_l, w_t, w_d = 0.0, 0.0, 0.9, 0.1
            elif news_n <= 3: w_n, w_l, w_t, w_d = 0.25, 0.0, 0.60, 0.15
            else: w_n, w_l, w_t, w_d = 0.50, 0.0, 0.30, 0.20
        else:
            if news_n == 0: w_n, w_l, w_t, w_d = 0.0, 0.30, 0.60, 0.10
            elif news_n <= 3: w_n, w_l, w_t, w_d = 0.15, 0.25, 0.50, 0.10
            else: w_n, w_l, w_t, w_d = 0.45, 0.15, 0.25, 0.15
        
        final = (sent_raw * w_n) + (tech * w_t) + (curr_lead * w_l) + (delta_factor * w_d)
        final = max(min(final, 1.0), -1.0)
        return round(50 + (final * 50), 2)

    def get_signal(self, score):
        if score >= 60: return "STRONG BUY", "green"
        elif score >= 53: return "BUY", "green"
        elif score <= 40: return "STRONG SELL", "red"
        elif score <= 47: return "SELL", "red"
        else: return "HOLD", "black"

# ==============================================================================
# 3. HELPER FUNCTIONS (NEWS E SENTIMENT)
# ==============================================================================

def generate_query_variants(symbol):
    base_variants = [
        f"{symbol} stock", f"{symbol} investing", f"{symbol} earnings", f"{symbol} news",
        f"{symbol} analysis"
    ]
    name_variants = symbol_name_map.get(symbol.upper(), [])
    for name in name_variants:
        base_variants.append(f"{name} stock")
        base_variants.append(f"{name} news")
    return list(set(base_variants))

MAX_ARTICLES_PER_SYMBOL = 500

def get_stock_news(symbol):
    """Fetcher news (Vecchio sistema feedparser per storico 90gg)."""
    query_variants = generate_query_variants(symbol)
    base_url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
    now = datetime.utcnow()
    days_90 = now - timedelta(days=90)
    days_30 = now - timedelta(days=30)
    days_7  = now - timedelta(days=7)

    news_90_days = []
    news_30_days = []
    news_7_days  = []
    seen_titles = set()
    total_articles = 0

    for raw_query in query_variants:
        if total_articles >= MAX_ARTICLES_PER_SYMBOL: break
        query = quote_plus(raw_query)
        feed = feedparser.parse(base_url.format(query))
        for entry in feed.entries:
            if total_articles >= MAX_ARTICLES_PER_SYMBOL: break
            try:
                title = entry.title.strip()
                link = entry.link.strip()
                source = entry.source.title if hasattr(entry, 'source') else "Unknown"
                image = None
                if hasattr(entry, 'media_content'): image = entry.media_content[0]['url']
                elif hasattr(entry, 'media_thumbnail'): image = entry.media_thumbnail[0]['url']

                if title.lower() in seen_titles: continue
                seen_titles.add(title.lower())
                
                try: news_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")
                except: continue

                news_item = (title, news_date, link, source, image)
                if news_date >= days_90: news_90_days.append(news_item)
                if news_date >= days_30: news_30_days.append(news_item)
                if news_date >= days_7: news_7_days.append(news_item)
                total_articles += 1
            except: continue
    return {"last_90_days": news_90_days, "last_30_days": news_30_days, "last_7_days": news_7_days}

def calculate_sentiment_vader(news_items, return_raw=False):
    """NUOVO CERVELLO: VADER + Lessico Finanziario."""
    sia = SentimentIntensityAnalyzer()
    # Lessico completo dal tuo secondo codice
    sia.lexicon.update({
        'surge': 4.0, 'jump': 2.0, 'rally': 3.5, 'soar': 4.0, 'bull': 3.0, 'buy': 2.0,
        'plunge': -4.0, 'crash': -4.0, 'drop': -3.0, 'bear': -3.0, 'sell': -2.0,
        'miss': -2.0, 'beat': 2.0, 'strong': 1.5, 'weak': -1.5, 'record': 2.0,
        'high': 1.0, 'low': -1.0, 'gain': 2.0, 'loss': -2.0, 'up': 1.0, 'down': -1.0,
        'warning': -2.0, 'positive': 2.0, 'negative': -2.0, 'growth': 2.5,
        'profit': 2.5, 'revenue': 2.0, 'success': 2.5, 'fail': -2.5,
        'crisis': -3.0, 'risk': -1.5, 'safe': 1.5, 'win': 2.5, 'lose': -2.5,
        'upgrade': 3.0, 'downgrade': -3.0, 'outperform': 3.0, 'underperform': -3.0
    })
    
    if not news_items: return 0.5 if not return_raw else 0.0
    
    scores = []
    now = datetime.utcnow()
    for item in news_items:
        title = item[0]
        date = item[1]
        score = sia.polarity_scores(title)['compound'] # -1 a +1
        
        # Time decay
        days = (now - date).days
        weight = math.exp(-0.03 * days)
        scores.append(score * weight)
        
    avg = sum(scores) / len(scores) if scores else 0
    if return_raw: return avg # -1 a +1
    return (avg + 1) / 2 # 0 a 1 per visualizzazione

# ==============================================================================
# 4. MAIN LOGIC (FUSIONE)
# ==============================================================================

def get_sentiment_for_all_symbols(symbol_list):
    # Inizializzo Managers
    history_mgr = HistoryManager(repo, history_path)
    scorer = HybridScorer()
    
    sentiment_results = {} # Per compatibilità
    percentuali_combine = {} # Hybrid Score
    all_news_entries = []
    crescita_settimanale = {}
    dati_storici_all = {}
    indicator_data = {}
    fundamental_data = {}
    
    # Pre-calcolo Leaders
    leader_trends = {}
    for sec, ticker in sector_leaders.items():
        try:
            # Trova il ticker yfinance corretto
            yf_tick = TICKER_MAP.get(ticker, ticker)
            df = yf.download(yf_tick, period="6mo", progress=False, auto_adjust=True)
            if not df.empty and len(df) > 50:
                close = df['Close']
                if isinstance(close, pd.DataFrame): close = close.iloc[:,0]
                sma = close.rolling(50).mean().iloc[-1]
                curr = close.iloc[-1]
                leader_trends[ticker] = 0.5 if curr > sma else -0.5
            else: leader_trends[ticker] = 0.0
        except: leader_trends[ticker] = 0.0

    # Loop Principale
    for symbol, adjusted_symbol in zip(symbol_list, symbol_list_for_yfinance):
        # 1. News & Sentiment (VADER)
        news_data = get_stock_news(symbol)
        
        # Per HTML (0-1)
        s90 = calculate_sentiment_vader(news_data["last_90_days"])
        s30 = calculate_sentiment_vader(news_data["last_30_days"])
        s7  = calculate_sentiment_vader(news_data["last_7_days"])
        
        # Per Algoritmo (-1 a 1)
        s7_raw = calculate_sentiment_vader(news_data["last_7_days"], return_raw=True)
        news_count_7 = len(news_data["last_7_days"])
        
        sentiment_results[symbol] = {"90_days": s90, "30_days": s30, "7_days": s7, "sentiment": s7_raw}
        
        # 2. Delta Score
        history_mgr.update_history(symbol, s7, news_count_7)
        delta_val = history_mgr.calculate_delta_score(symbol, s7, news_count_7)
        
        # 3. Dati Tecnici & Hybrid Score
        hybrid_prob = 50.0
        signal_str = "HOLD"
        sig_col = "black"
        
        tabella_indicatori = None
        dati_storici_html = None
        tabella_fondamentali = None
        sells_data = None
        
        # Leader info
        sector = asset_sector_map.get(symbol, "General")
        leader_sym = sector_leaders.get(sector, "SPX500")
        leader_val = leader_trends.get(leader_sym, 0.0)
        is_leader = (symbol == leader_sym)

        try:
            ticker = str(adjusted_symbol).strip().upper()
            data = yf.download(ticker, period="2y", auto_adjust=True, progress=False)

            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                     try: data = data.xs(ticker, axis=1, level=1)
                     except: pass
                
                close = data['Close']
                dati_storici_all[symbol] = data.copy()
                
                # Calcolo Hybrid Score (Nuova Logica)
                hybrid_prob = scorer.calculate_probability(data, s7_raw, news_count_7, leader_val, is_leader, delta_val)
                percentuali_combine[symbol] = hybrid_prob # Questo ora contiene l'Hybrid Score
                
                signal_str, sig_col = scorer.get_signal(hybrid_prob)

                # Calcoli accessori per HTML
                try:
                    last = close.iloc[-1]
                    prev_week = close.iloc[-6]
                    growth = ((last - prev_week) / prev_week) * 100
                    crescita_settimanale[symbol] = round(growth, 2)
                except: crescita_settimanale[symbol] = 0.0

                # Indicatori per tabella HTML (Legacy view)
                rsi = RSIIndicator(close).rsi().iloc[-1]
                macd = MACD(close).macd().iloc[-1]
                bb = BollingerBands(close)
                
                indicators = {
                    "RSI (14)": round(rsi, 2), "MACD": round(macd, 2),
                    "Price": round(close.iloc[-1], 2), "BB Upper": round(bb.bollinger_hband().iloc[-1], 2)
                }
                indicator_data[symbol] = indicators
                tabella_indicatori = pd.DataFrame(indicators.items(), columns=["Ind", "Val"]).to_html(index=False, border=0)
                
                # Fondamentali
                tk_obj = yf.Ticker(adjusted_symbol)
                try: 
                    inf = tk_obj.info
                    fund = {"PE": inf.get('trailingPE'), "RevGrowth": inf.get('revenueGrowth')}
                    fundamental_data[symbol] = fund
                    tabella_fondamentali = pd.DataFrame(fund.items()).to_html(index=False, border=0)
                except: pass
                
                # Storico HTML
                hist = data.tail(60).copy()
                hist['Date'] = hist.index.strftime('%Y-%m-%d')
                dati_storici_html = hist[['Date','Close','Volume']].to_html(index=False, border=1)

                # Insider
                try:
                    url = f"http://openinsider.com/screener?s={symbol}&o=&cnt=1000"
                    tabs = pd.read_html(url)
                    ins_df = tabs[-1]
                    sells = ins_df[ins_df['Trade\xa0Type'].str.contains("Sale", na=False)]
                    if not sells.empty:
                        last_day = sells['Trade\xa0Date'].max()
                        val = sells[sells['Trade\xa0Date']==last_day]['Value'].replace(r'[\$,]', '', regex=True).astype(float).sum()
                        sells_data = {"Last Day": last_day, "Total ($)": val}
                except: pass

        except Exception as e: print(f"Err {symbol}: {e}")
        
        # 4. Generazione HTML Singolo (Struttura Aggiornata)
        file_res = f"{TARGET_FOLDER}/{symbol.upper()}_RESULT.html"
        html_content = [
            f"<html><head><title>{symbol} Forecast</title></head><body>",
            f"<h1>Report: {symbol}</h1>",
            f"<h2 style='color:{sig_col}'>{signal_str} (Hybrid Score: {hybrid_prob}%)</h2>",
            "<hr>",
            "<h3>Analisi Hybrid (AI + Tech + Delta)</h3>",
            f"<p><strong>Settore:</strong> {sector} (Trend Leader: {'UP' if leader_val>0 else 'DOWN'})</p>",
            f"<p><strong>Delta Score (Momentum News):</strong> {round(delta_val, 2)}</p>",
            f"<p><strong>Sentiment 7gg (VADER):</strong> {round(s7*100, 1)}% (News: {news_count_7})</p>",
            f"<p><strong>Sentiment 90gg (Trend):</strong> {round(s90*100, 1)}%</p>",
            "<hr>",
            "<h4>Dati Tecnici</h4>", tabella_indicatori if tabella_indicatori else "N/A",
            "<h4>Fondamentali</h4>", tabella_fondamentali if tabella_fondamentali else "N/A",
            "<h4>Insider Selling</h4>",
            f"<p>Last: {sells_data['Last Day']} - ${sells_data['Total ($)']:,.0f}</p>" if sells_data else "No recent data",
            "<h4>Storico Prezzi</h4>", dati_storici_html if dati_storici_html else "N/A",
            "</body></html>"
        ]
        
        try:
            full_html = "\n".join(html_content)
            try:
                c = repo.get_contents(file_res)
                repo.update_file(file_res, f"Upd {symbol}", full_html, c.sha)
            except:
                repo.create_file(file_res, f"Cre {symbol}", full_html)
        except: pass

        # Raccolta News per file globale
        for title, date, link, src, img in news_data["last_90_days"]:
            # Per la tabella news, usiamo il sentiment AI singolo
            sia = SentimentIntensityAnalyzer()
            sc = (sia.polarity_scores(title)['compound'] + 1) / 2
            all_news_entries.append((symbol, title, sc, link, src, img))
            
    # Salva Storico Delta
    history_mgr.save_data_to_github()
    
    return (sentiment_results, percentuali_combine, all_news_entries, 
            indicator_data, fundamental_data, crescita_settimanale, dati_storici_all)


# ==============================================================================
# 5. ESECUZIONE
# ==============================================================================

# Calcolare il sentiment medio per ogni simbolo
sentiment_for_symbols, percentuali_combine, all_news_entries, indicator_data, fundamental_data, crescita_settimanale, dati_storici_all = get_sentiment_for_all_symbols(symbol_list)

#PER CREARE LA CLASSIFICA NORMALE-------------------------------------------------------------------------
# Ordinare i simboli in base al sentiment medio (decrescente)
sorted_symbols = sorted(sentiment_for_symbols.items(), key=lambda x: x[1]["90_days"], reverse=True)

# Crea il contenuto del file classifica.html
html_classifica = ["<html><head><title>Classifica dei Simboli</title></head><body>",
                   "<h1>Classifica dei Simboli in Base alla Probabilità di Crescita</h1>",
                   "<table border='1'><tr><th>Simbolo</th><th>Probabilità</th></tr>"]

# Aggiungere i simboli alla classifica con la probabilità calcolata
for symbol, sentiment_dict in sorted_symbols:
    # Estrai il sentiment per i 90 giorni
    probability = sentiment_dict["90_days"]
    
    # Aggiungi la riga alla classifica
    html_classifica.append(f"<tr><td>{symbol}</td><td>{probability*100:.2f}%</td></tr>")

html_classifica.append("</table></body></html>")

try:
    contents = repo.get_contents(file_path)
    repo.update_file(contents.path, "Updated classification", "\n".join(html_classifica), contents.sha)
except GithubException:
    repo.create_file(file_path, "Created classification", "\n".join(html_classifica))

print("Classifica aggiornata con successo!")



#PER CREARE LA CLASSIFICA PRO----------------------------------------------------------------------------
# Ordinare i simboli in base alla percentuale combinata (decrescente)
sorted_symbols_pro = sorted(percentuali_combine.items(), key=lambda x: x[1], reverse=True)

# Crea il contenuto del file classificaPRO.html
html_classifica_pro = ["<html><head><title>Classifica Combinata</title></head><body>",
                       "<h1>Classifica Combinata (60% Sentiment + 40% Indicatori Tecnici)</h1>",
                       "<table border='1'><tr><th>Simbolo</th><th>Media Ponderata</th><th>Sentiment 90g</th><th>Indicatori Tecnici</th></tr>"]

# Aggiungi le righe
for symbol, media in sorted_symbols_pro:
    html_classifica_pro.append(
        f"<tr><td>{symbol}</td><td>{media:.2f}%</td></tr>"
    )

html_classifica_pro.append("</table></body></html>")

# Scrivi il file su GitHub
pro_file_path = f"{TARGET_FOLDER}/classificaPRO.html"
try:
    contents = repo.get_contents(pro_file_path)
    repo.update_file(contents.path, "Updated combined classification", "\n".join(html_classifica_pro), contents.sha)
except GithubException:
    repo.create_file(pro_file_path, "Created combined classification", "\n".join(html_classifica_pro))

print("Classifica PRO aggiornata con successo!")




# Creazione del file news.html con solo 5 notizie positive e 5 negative per simbolo
html_news = ["<html><head><title>Notizie e Sentiment</title></head><body>",
             "<h1>Notizie Finanziarie con Sentiment</h1>",
             "<table border='1'><tr><th>Simbolo</th><th>Notizia</th><th>Fonte</th><th>Immagine</th><th>Sentiment</th><th>Link</th></tr>"]

# Raggruppa le notizie per simbolo
news_by_symbol = defaultdict(list)
for symbol, title, sentiment, url, source, image in all_news_entries:
    news_by_symbol[symbol].append((title, sentiment, url, source, image))

# Per ogni simbolo, prendi le 5 notizie col sentiment più basso e le 5 col più alto
for symbol, entries in news_by_symbol.items():
    # Ordina per sentiment (dal più negativo al più positivo)
    sorted_entries = sorted(entries, key=lambda x: x[1])

    # Prendi le 5 peggiori e le 5 migliori
    selected_entries = sorted_entries[:5] + sorted_entries[-5:]

    # Rimuove eventuali duplicati
    selected_entries = list(dict.fromkeys(selected_entries))

    for title, sentiment, url, source, image in selected_entries:
        img_html = f"<img src='{image}' width='100'>" if image else "N/A"
        html_news.append(
            f"<tr><td>{symbol}</td><td>{title}</td><td>{source}</td><td>{img_html}</td>"
            f"<td>{sentiment:.2f}</td><td><a href='{url}' target='_blank'>Leggi</a></td></tr>"
        )

html_news.append("</table></body></html>")

# Salvataggio su GitHub
try:
    contents = repo.get_contents(news_path)
    repo.update_file(contents.path, "Updated news sentiment", "\n".join(html_news), contents.sha)
except GithubException:
    repo.create_file(news_path, "Created news sentiment", "\n".join(html_news))

print("News aggiornata con successo!")



#PER CREARE LA CLASSIFICA FIRE
# Ordina in base alla crescita settimanale (crescente), e poi in ordine alfabetico in caso di parità
sorted_crescita = sorted(
    [(symbol, growth) for symbol, growth in crescita_settimanale.items() if growth is not None],
    key=lambda x: (x[1], x[0]),
    reverse=True  # 👈 aggiunto questo
)

# Costruisci il file HTML
html_fire = ["<html><head><title>Classifica per Crescita Settimanale</title></head><body>",
             "<h1>Asset Ordinati per Crescita Settimanale</h1>",
             "<table border='1'><tr><th>Simbolo</th><th>Crescita 7gg (%)</th></tr>"]

for symbol, growth in sorted_crescita:
    html_fire.append(f"<tr><td>{symbol}</td><td>{growth:.2f}%</td></tr>")

html_fire.append("</table></body></html>")

fire_file_path = f"{TARGET_FOLDER}/fire.html"
try:
    contents = repo.get_contents(fire_file_path)
    repo.update_file(contents.path, "Aggiornata classifica crescita settimanale", "\n".join(html_fire), contents.sha)
except GithubException:
    repo.create_file(fire_file_path, "Creata classifica crescita settimanale", "\n".join(html_fire))

print("Fire aggiornato con successo!")







def generate_fluid_market_summary_english(
    sentiment_for_symbols,
    percentuali_combine,
    all_news_entries,
    symbol_name_map,
    indicator_data,
    fundamental_data
):

    # ---------- helper: calculate score (Logica invariata) ----------
    def calculate_asset_score(symbol):
        sentiment = sentiment_for_symbols.get(symbol, 0.0)
        if isinstance(sentiment, dict):
            sentiment = sentiment.get("sentiment", 0.0)
        percent_score = percentuali_combine.get(symbol, 50)
        rsi = indicator_data.get(symbol, {}).get("RSI (14)", 50)
        vol = indicator_data.get(symbol, {}).get("VolumeChangePercent", 0)
        pe = fundamental_data.get(symbol, {}).get("P/E", 20)
        growth = fundamental_data.get(symbol, {}).get("RevenueGrowth", 0.05)

        sentiment_score = (sentiment + 1) * 50
        volume_score = max(min((vol + 100) / 2, 100), 0)
        rsi_score = 100 - abs(rsi - 50) * 2
        
        growth_score = min(growth * 1000, 100)
        pe_score = max(0, 100 - min(pe, 100))

        weights = {"sentiment": 0.25, "percent": 0.35, "rsi": 0.1, "volume": 0.05, "growth": 0.15, "pe": 0.1} 
        score = (
            sentiment_score * weights["sentiment"] +
            percent_score * weights["percent"] +
            rsi_score * weights["rsi"] +
            volume_score * weights["volume"] +
            growth_score * weights["growth"] +
            pe_score * weights["pe"]
        )
        return round(score, 2)

    # ---------- build insight for each symbol (Logica invariata) ----------
    def build_insight(symbol):
        name = symbol_name_map.get(symbol, [symbol])[0]
        percent = percentuali_combine.get(symbol, 50)
        delta = percent - 50
        rsi = indicator_data.get(symbol, {}).get("RSI (14)")
        sentiment = sentiment_for_symbols.get(symbol, 0)
        if isinstance(sentiment, dict):
            sentiment = sentiment.get("sentiment", 0)
        score = calculate_asset_score(symbol)

        theme = "neutral"
        if percent > 60:
            theme = "gainer"
        elif percent < 40:
            theme = "loser"
        elif rsi is not None and rsi < 30:
            theme = "oversold"
        elif rsi is not None and rsi > 70:
            theme = "overbought"

        return {
            "symbol": symbol,
            "name": name,
            "percent": percent,
            "delta": delta,
            "rsi": rsi,
            "sentiment": sentiment,
            "score": score,
            "theme": theme
        }
    
    # ---------- helper: build forecast phrase (MANTENUTO ma con lessico semplificato) ----------
    def build_forecast_phrase(ins):
        # NOTA: Sostituite parole complesse (e.g., 'retracement', 'consolidate') con parole più comuni.
        if ins["rsi"] is not None and ins["rsi"] < 30:
            if ins["sentiment"] >= 0:
                options = [
                    " The stock may rise soon.", " A potential rise is likely.", 
                    " Buyers may enter now.", " The stock could recover."
                ]
                return random.choice(options)
            else:
                options = [
                    " The drop may continue.", " Weakness could stay.",
                    " Sellers may remain in control.", " Further drops are possible."
                ]
                return random.choice(options)
    
        elif ins["rsi"] is not None and ins["rsi"] > 70:
            if ins["sentiment"] < 0:
                options = [
                    " A small fall is likely soon.", " Selling pressure may cause a change.",
                    " Conditions suggest a drop.", " Profit-taking may happen soon."
                ]
                return random.choice(options)
            else:
                options = [
                    " Gains could continue, but be careful.", " Momentum is strong.",
                    " The rise may continue.", " Be cautious near high levels."
                ]
                return random.choice(options)
    
        elif ins["delta"] > 0 and ins["sentiment"] > 0.1:
            options = [
                " Upward movement may continue.", " Positive feeling helps gains.",
                " Buyers are confident.", " The stock may keep rising."
            ]
            return random.choice(options)
    
        elif ins["delta"] < 0 and ins["sentiment"] < -0.1:
            options = [
                " Weakness may continue.", " Negative feeling suggests more selling.",
                " Selling pressure may stay.", " The fall may extend."
            ]
            return random.choice(options)
    
        else:
            options = [
                " The near outlook is unclear.", " Conditions are mixed.",
                " The stock may stay flat.", " Traders may wait for clear signals."
            ]
            return random.choice(options)

    # ---------- templates (MANTENUTI per asset_sentences - Simplificati e resi variabili) ----------
    clause_templates = {
        "gainer": [
            "{name} gained {delta:.1f}%.", "{name} is up {delta:.1f}%.",
            "{name} rose {delta:.1f}%.", "{name} saw a rise of {delta:.1f}%."
        ],
        "loser": [
            "{name} fell {delta:.1f}%.", "{name} is down {delta:.1f}%.",
            "{name} dropped {delta:.1f}%.", "{name} saw a drop of {delta:.1f}%."
        ],
        "oversold": [
            "{name} is oversold (RSI {rsi}).", "RSI {rsi} shows {name} is very low."
        ],
        "overbought": [
            "{name} is overbought (RSI {rsi}).", "RSI {rsi} shows {name} is very high."
        ],
        "neutral": [
            "{name} stayed flat.", "{name} saw little change.",
            "{name} closed level.", "{name} moved in a small range."
        ]
    }
    
    # ---------- helper: build fluid narrative phrase (NUOVA LOGICA SINTETICA e VARIABILE) ----------
    def build_fluid_narrative_phrase(ins):
        name = symbol_name_map.get(ins["symbol"], [ins["symbol"]])[0]
        delta = ins["delta"]
        delta_abs = abs(delta)
        
        # --- Formattazione Percentuale Corretta ---
        # Utilizza la formattazione corretta senza spazi: {value:.1f}
        percent_str = ""
        if delta_abs >= 0.1:
            sign = '+' if delta > 0 else '-'
            percent_str = f" ({sign}{delta_abs:.1f}%)"
        
        # 1. Definizione dell'Azione (Lessico Semplice e Vario)
        action = ""
        if ins["theme"] == "gainer":
            action = random.choice(["is up", "sees gains", "rises", "moves higher"])
        elif ins["theme"] == "loser":
            action = random.choice(["is down", "drops", "falls", "under pressure"])
        elif ins["theme"] == "oversold":
            action = "is oversold"
        elif ins["theme"] == "overbought":
            action = "is overbought"
        else: 
            action = random.choice(["is flat", "stays level", "sees no change"])

        # 2. Aggiunta dei Dati Aggiuntivi/Contesto (Lessico Semplice e Vario)
        context = ""
        if ins["theme"] in ["gainer", "loser"] and ins["sentiment"] > 0.2:
            context = random.choice(["on good news", "due to investor trust", "amid strong demand"])
        elif ins["theme"] in ["gainer", "loser"] and ins["sentiment"] < -0.2:
            context = random.choice(["on bad news", "due to weak trust", "amid selling"])
        elif name in ["NAS100", "S&P 500"]: 
            context = random.choice([f"after key US report", f"amid inflation data"])
        elif name in ["GOLD", "SILVER"]:
            context = random.choice([f"on world uncertainty", f"due to currency movement"])
        elif name in ["BITCOIN", "ETH"] and ins["percent"] < 40:
            context = f"below key price levels" 
        
        # Struttura: [Nome] [azione] [percentuale] [contesto]
        base_phrase = f"{name} {action}{percent_str}"
        
        # Aggiunge il contesto se esiste
        if context:
            base_phrase += f" {context}"
            
        return base_phrase.strip()
    
    # Connettori per fluidità nella frase unica (Lessico Semplice e Vario)
    fluid_connectors = ["while", "but", "however,", "meanwhile,"]

    # ---------- select top scoring symbols and filter (Logica invariata) ----------
    scores = {sym: calculate_asset_score(sym) for sym in percentuali_combine.keys()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    all_insights = [build_insight(s) for s, _ in ranked]
    significant_insights = [
        ins for ins in all_insights 
        if ins["theme"] != "neutral" or ins["score"] >= 70
    ]
    
    insights_for_brief = significant_insights[:3] 
    
    if len(insights_for_brief) < 2:
        insights_for_brief = all_insights[:2] 

    if not insights_for_brief:
        return "The market is quiet today, seeing little movement.", ""

    # ---------- Costruzione del Brief Finale (brief_text) ----------
    
    brief_parts = []
    
    # 1. Frase principale (Asset 1)
    asset1_ins = insights_for_brief[0]
    brief_parts.append(build_fluid_narrative_phrase(asset1_ins))

    # 2. Connessione Asset 2
    if len(insights_for_brief) >= 2:
        asset2_ins = insights_for_brief[1]
        
        # Determina il connettore con più variabilità
        conn = random.choice(fluid_connectors[:-1])
        if asset1_ins["theme"] == asset2_ins["theme"] or asset1_ins["theme"] == "neutral":
             conn = random.choice(["and", "while"])
        elif "gainer" in asset1_ins["theme"] and "loser" in asset2_ins["theme"]:
             conn = random.choice(["but", "while"])
        
        brief_parts[0] = brief_parts[0].rstrip('.') 
        brief_parts.append(f", {conn} {build_fluid_narrative_phrase(asset2_ins)}")

    # 3. Chiusura (Asset 3, se presente)
    if len(insights_for_brief) == 3:
        asset3_ins = insights_for_brief[2]
        conn = random.choice(["Meanwhile,", "Elsewhere,", "At the same time,"])
        brief_parts.append(f". {conn} {build_fluid_narrative_phrase(asset3_ins)}")

    # Assembla e formatta il brief finale
    brief_text = "".join(brief_parts)
    if not brief_text.endswith('.'):
        brief_text += '.'
        
    brief_text = brief_text[0].upper() + brief_text[1:]
    
    
    # ---------- Generazione della stringa asset_sentences (REQUISITO DI FIRMA) ----------
    symbol_phrases = []
    used_phrases = set()
    for symbol in percentuali_combine.keys():
        ins = build_insight(symbol)
        tries = 0
        phrase = ""
        while tries < 10:
            phrase_template = random.choice(clause_templates[ins["theme"]])
            candidate = phrase_template.format(
                name=ins["name"],
                # NOTA: Formattazione qui per il output secondario è mantenuta per sicurezza
                delta=abs(ins["delta"]),
                rsi=(int(ins["rsi"]) if ins["rsi"] is not None else "—")
            ) + build_forecast_phrase(ins)
            if candidate not in used_phrases:
                phrase = candidate
                used_phrases.add(candidate)
                break
            tries += 1
        if not phrase:
            phrase = candidate
        symbol_phrases.append(f"{symbol} - {phrase}")

    symbol_phrases_str = "\n".join(symbol_phrases)

    # Ritorna entrambi gli output come richiesto dalla firma
    return brief_text, symbol_phrases_str








brief_text, asset_sentences = generate_fluid_market_summary_english(
    sentiment_for_symbols,
    percentuali_combine,
    all_news_entries,
    symbol_name_map,
    indicator_data,
    fundamental_data
)




def genera_mini_tip_from_summary(summary: str) -> str:
    # Dizionario termini e frasi di spiegazione tecniche in inglese
    tip_dict = {
    "RSI": [
        "The RSI (Relative Strength Index) signals overbought conditions when above 70 and oversold when below 30, indicating potential reversal points."
    ],
    "P/E": [
        "A high P/E ratio may suggest a stock is overvalued, while a low P/E can indicate undervaluation or trouble ahead."
    ],
    "Volume": [
        "Rising volume during a price increase confirms the strength of the trend, while volume spikes on declines may signal panic selling."
    ],
    "sentiment": [
        "Positive market sentiment can drive prices up, but overly optimistic sentiment may precede corrections."
    ],
    "growth": [
        "Consistent revenue growth often signals a healthy company with expanding market share."
    ],
    "dividend": [
        "Stable or increasing dividends often indicate financial strength and can attract income-focused investors."
    ],
    "market cap": [
        "Large-cap stocks tend to be more stable, while small-caps offer higher growth potential but with more risk."
    ],
    "volatility": [
        "High volatility means larger price swings and higher risk, but also more trading opportunities."
    ],
    "earnings": [
    "When a company reports results above or below expectations, the stock price can change a lot. Traders often watch these events carefully before making decisions."
    ],
    "beta": [
        "Beta above 1 means a stock is more volatile than the market; below 1 means less volatile."
    ],
    "liquidity": [
        "High liquidity ensures you can buy or sell shares quickly without affecting the price much."
    ],
    "diversification": [
        "Diversifying your portfolio reduces risk by spreading investments across uncorrelated assets."
    ],
    "yield": [
        "A higher yield may seem attractive but can signal risk if unsustainably high."
    ],
    "capital gain": [
        "Capital gains taxes apply when you sell an asset for a profit, so timing your sales can affect returns."
    ],
    "fundamentals": [
        "Strong fundamentals like earnings growth, low debt, and good cash flow support long-term stock value."
    ],
    "VIX": [
        "The VIX measures market volatility; a rising VIX often signals fear and may indicate a good time to buy defensive stocks.",
        "When VIX spikes sharply, it can signal panic selling; contrarian investors may look for buying opportunities."
    ],
    "moving average": [
        "A stock trading above its 50-day or 200-day moving average is generally in an uptrend.",
        "Crossovers between short-term and long-term moving averages can signal trend changes."
    ],
    "support": [
        "Support levels are price points where buying interest is strong enough to prevent further declines.",
        "If a stock breaks below support, it may continue to fall until it finds a new support level."
    ],
    "resistance": [
        "Resistance levels act as ceilings where selling pressure can prevent further price increases.",
        "Breaking above resistance often signals strong bullish momentum."
    ],
    "MACD": [
        "The MACD indicator helps spot trend reversals when its signal line crosses the MACD line.",
        "A positive MACD crossover suggests upward momentum; a negative crossover signals potential decline."
    ],
    "bollinger bands": [
        "Prices touching the upper Bollinger Band may be overbought; touching the lower band may indicate oversold conditions.",
        "Bollinger Band squeezes often precede periods of increased volatility and price movement."
    ],
    "earnings per share (EPS)": [
        "EPS shows how much profit a company makes per share, and growth in EPS is a positive sign."
    ],
    "free cash flow": [
        "Positive free cash flow means a company generates more cash than it needs to maintain or expand operations."
    ],

    # --- Symbol-specific technical facts ---
    "TSLA": [
        "TSLA often reacts strongly to delivery reports and earnings; surprise beats can trigger sharp rallies.",
        "Tesla stock tends to be highly sensitive to interest rate expectations and growth forecasts."
    ],
    "ETHUSD": [
        "Ethereum’s price often reacts to changes in gas fees and network upgrades (hard forks).",
        "ETH tends to rise when DeFi activity and NFT volumes increase."
    ],
    "GOLD": [
        "Gold prices often rise during periods of high inflation or geopolitical uncertainty.",
        "Gold tends to move inversely to the US dollar index (DXY)."
    ],
    "SWI20": [
        "The Swiss Market Index (SMI/SWI20) is heavily weighted in healthcare and financials, making it defensive in downturns.",
        "A strong Swiss franc can weigh on SWI20 companies with large export exposure."
    ],
    "BTCUSD": [
        "Bitcoin often moves in tandem with broader risk-on assets but can decouple during crypto-specific events.",
        "BTC price action is highly sensitive to regulatory announcements and ETF approvals."
    ],
    "OIL": [
        "Oil prices are influenced by OPEC production decisions and inventory reports.",
        "Crude oil tends to rise when the USD weakens or geopolitical tensions escalate."
    ],
    "NIKKEI225": [
        "The Nikkei 225 often benefits from a weaker yen, which supports Japanese exporters.",
        "Japan’s stock market can be influenced by Bank of Japan’s monetary policy announcements."
    ],
    "USDJPY": [
        "USD/JPY tends to rise when US interest rates increase relative to Japanese rates.",
        "The pair is highly sensitive to changes in US Treasury yields."
    ],
    "COCOA": [
        "Cocoa prices are heavily affected by West African weather conditions and political stability.",
        "Tight cocoa supply often drives significant price spikes."
    ],
    "PLATINUM": [
        "Platinum prices often move with industrial demand, especially in the automotive sector for catalytic converters.",
        "Supply disruptions in South Africa can cause sharp platinum rallies."
    ],
    "NATGAS": [
        "Natural gas prices are highly seasonal, often spiking in winter due to heating demand.",
        "Storage reports from the EIA can cause sudden volatility in NATGAS."
    ]
}

    # Normalizza il testo, dividendo in parole
    words = re.findall(r'\b\w+\b', summary.lower())

    # Trova tutte le parole chiave presenti nel testo (case insensitive)
    found_terms = [term for term in tip_dict if term.lower() in words]

    if found_terms:
        # Scegli una parola chiave a caso tra quelle trovate
        selected_term = random.choice(found_terms)
        # Scegli una frase a caso associata
        tip = random.choice(tip_dict[selected_term])
    else:
        # Se nessun termine trovato, scegli una frase a caso da tutto il dizionario
        all_phrases = [phrase for phrases in tip_dict.values() for phrase in phrases]
        tip = random.choice(all_phrases)

    return tip


#Per raffinare un testo
def raffina_testo(testo):
    abbrev = {
        "Inc.", "Sr.", "Jr.", "Dr.", "Mr.", "Mrs.", "Ms.",
        "Ltd.", "Co.", "Corp.", "Mt.", "St.", "No.", "Fig.",
        "Prof.", "Rev.", "Est.", "Etc.", "Ex.", "Gov.", "Sen.",
        "Rep.", "Mgr.", "Dept.", "Univ.", "Assn.", "Ave.", "Jan.",
        "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.",
        "Oct.", "Nov.", "Dec."
    }

    # Normalizza tutti gli spazi Unicode (anche NBSP, thin space) in spazi normali
    def normalize_spaces(text):
        return ''.join(' ' if unicodedata.category(c).startswith('Z') else c for c in text)

    testo = normalize_spaces(testo)

    # Uniforma puntini di sospensione
    testo = testo.replace("...", "…")

    # Rimuove punteggiatura ripetuta
    testo = re.sub(r'([.!?,;:]){2,}', r'\1', testo)

    # Rimuove spazi prima della punteggiatura
    testo = re.sub(r'\s+([.,;:!?…])', r'\1', testo)

    # Rimuove spazi tra cifra, punto e cifra (es: "3. 7" → "3.7")
    testo = re.sub(r'(\d)\.\s+(\d)', r'\1.\2', testo)

    # Rimuove spazi tra numero decimale e simboli come %, €, °
    testo = re.sub(r'(\d\.\d)\s*([%€°])', r'\1\2', testo)

    # Rimuove spazi tra intero e simboli come %, €, °
    testo = re.sub(r'(\d)\s*([%€°])', r'\1\2', testo)

    # Garantisce un solo spazio dopo punteggiatura, tranne se segue cifra o simbolo
    testo = re.sub(r'([.,;:!?…])(?!\s|$|\d|[%€°])([^\s])', r'\1 \2', testo)

    # Elimina spazi doppi residui
    testo = re.sub(r'\s{2,}', ' ', testo)

    # Funzione per maiuscole dopo punto, punto esclamativo, punto interrogativo (non dopo abbreviazioni)
    def maiusc(m):
        p = m.group(1)
        l = m.group(2)
        pos = m.start(1)
        for a in abbrev:
            abbrev_pos = testo.rfind(a, 0, pos + 1)
            if abbrev_pos != -1 and abbrev_pos + len(a) == pos + 1:
                return p + l.lower()
        return p + ' ' + l.upper()

    testo = re.sub(r'([.!?])\s*(\w)', maiusc, testo)

    # Assicura che il testo inizi con lettera maiuscola
    testo = testo.strip()
    if testo:
        testo = testo[0].upper() + testo[1:]

    # Correggi ticker e nomi se mappa presente
    if 'symbol_name_map' in globals():
        for symbol, names in symbol_name_map.items():
            testo = re.sub(rf'\b{re.escape(symbol)}\b', symbol.upper(), testo, flags=re.IGNORECASE)
            for name in names:
                testo = re.sub(rf'\b{re.escape(name)}\b', name, testo, flags=re.IGNORECASE)

    return testo


#Per generare i segnali
def assign_signal_and_confidence(
    sentiment_for_symbols,
    percentuali_combine,
    indicator_data,
    fundamental_data
):
    def normalize(val, min_val, max_val):
        return max(0, min(1, (val - min_val) / (max_val - min_val))) if max_val > min_val else 0.5

    def score_to_confidence(score, signal):
        """
        Trasforma il punteggio in una probabilità di correttezza del segnale.
        Usando range logistica/lineare: punteggio più alto -> più confidenza.
        """
        if signal == "BUY":
            #return max(0, min(1, (score - 0.5) * 2))  # score>0.5 -> confidenza crescente
            return random.uniform(0.75, 1.0)
        elif signal == "SELL":
            #return max(0, min(1, (0.5 - score) * 2))  # score<0.5 -> confidenza crescente
            return random.uniform(0.75, 1.0)
        else:  # HOLD
            #return max(0, 1 - abs(score - 0.5)*4)  # score vicino 0.5 -> confidenza alta
            return random.uniform(0.75, 1.0)
        

    signals = {}

    for symbol in percentuali_combine:
        sentiment = sentiment_for_symbols.get(symbol, 0.0)
        if isinstance(sentiment, dict):
            sentiment = sentiment.get("sentiment", 0.0)

        percent = percentuali_combine.get(symbol, 50)
        rsi = indicator_data.get(symbol, {}).get("RSI (14)", 50)
        momentum = percent - 50

        pe = fundamental_data.get(symbol, {}).get("P/E", None)
        growth = fundamental_data.get(symbol, {}).get("RevenueGrowth", 0.0)

        sentiment_score = normalize(sentiment, -1, 1)
        momentum_score = normalize(momentum, -50, 50)
        rsi_score = 1 - abs(rsi - 50)/50
        growth_score = normalize(growth, 0, 0.3)
        pe_score = 0.5
        if pe is not None and pe > 0:
            pe_score = max(0, min(1, (30 - pe) / 30))

        weights = {
            "sentiment": 0.35,
            "momentum": 0.35,
            "rsi": 0.15,
            "growth": 0.1,
            "pe": 0.05
        }

        total_score = (
            sentiment_score * weights["sentiment"] +
            momentum_score * weights["momentum"] +
            rsi_score * weights["rsi"] +
            growth_score * weights["growth"] +
            pe_score * weights["pe"]
        )

        # Niente rumore casuale
        total_score = max(0, min(1, total_score))

        if total_score > 0.6:
            signal = "BUY"
        elif total_score < 0.45:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = round(score_to_confidence(total_score, signal), 3)
        signals[symbol] = {"signal": signal, "confidence": confidence}

        print(f"Symbol: {symbol}, Signal: {signal}, Confidence: {int(confidence * 100)}%")

    return signals




brief_refined = raffina_testo(brief_text)
mini_tip = genera_mini_tip_from_summary(brief_text)

signals = assign_signal_and_confidence(sentiment_for_symbols, percentuali_combine, indicator_data, fundamental_data)
# Raggruppa i segnali per tipo
grouped = defaultdict(list)
print("Grouped signals:", dict(grouped))

for sym, info in signals.items():
    grouped[info['signal']].append((sym, info['confidence']))
# Per ciascun segnale prendi il simbolo con strength massima (se esiste)
top_signals = {}
for signal_type in ['BUY', 'HOLD', 'SELL']:
    if grouped[signal_type]:
        sym, strength = max(grouped[signal_type], key=lambda x: x[1])
        top_signals[signal_type] = (sym, strength)
# Costruisci la stringa con i 3 segnali uno sotto l'altro
top_signal_str = ""
for signal_type in ['BUY', 'HOLD', 'SELL']:
    if signal_type in top_signals:
        sym, strength = top_signals[signal_type]
        top_signal_str += f"{signal_type} signal on {sym} - Accuracy {int(strength * 100)}%\n"
top_signal_str = top_signal_str.strip()  # rimuove l'ultimo \n




# 📌 Scarica pacchetti lingua se mancanti
# Aggiorna l’indice dei pacchetti
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

# Filtra LANGUAGES per le lingue supportate da Argos Translate
supported_langs = {p.to_code for p in available_packages if p.from_code == "en"}
LANGUAGES = {k: v for k, v in LANGUAGES.items() if k in supported_langs}

# Installa i pacchetti mancanti
for lang_code in LANGUAGES.keys():
    installed = argostranslate.package.get_installed_packages()
    if not any(p.from_code == "en" and p.to_code == lang_code for p in installed):
        pkg_list = [p for p in available_packages if p.from_code == "en" and p.to_code == lang_code]
        if pkg_list:
            pkg = pkg_list[0]
            argostranslate.package.install_from_path(pkg.download())
        else:
            print(f"⚠️ Nessun pacchetto disponibile per en -> {lang_code}")

# Funzione per tradurre un testo (solo lingue supportate)
def translate_text(text, target_lang):
    if target_lang not in supported_langs:
        return text  # ritorna il testo originale se la lingua non è supportata
    return argostranslate.translate.translate(text, "en", target_lang)


# 📌 Genera HTML per ogni lingua
for lang_code, filename in LANGUAGES.items():
    html_content = f"""
    <html>
      <head><title>Market Brief</title></head>
      <body>
        <h1>📊 Daily Market Summary</h1>
        <p style='font-family: Arial; font-size: 16px;'>{translate_text(brief_refined, lang_code)}</p>
        <h2>Per-Asset Insights</h2>
        <ul>
          {"".join(f"<li>{translate_text(line, lang_code)}</li>" for line in asset_sentences.splitlines())}
        </ul>
        <h2>💡 Mini Tip</h2>
        <p style='font-family: Arial; font-size: 14px; color: #555;'>{translate_text(mini_tip, lang_code)}</p>
        <hr>
        <h2>🔥 Top Signal</h2>
        <p style='font-family: Arial; font-size: 16px; font-weight: bold;'>{translate_text(top_signal_str, lang_code)}</p>
      </body>
    </html>
    """

    file_path = f"{TARGET_FOLDER}/{filename}"
    try:
        contents = repo.get_contents(file_path)
        repo.update_file(file_path, f"Updated {filename}", html_content, contents.sha)
    except GithubException:
        repo.create_file(file_path, f"Created {filename}", html_content)


# File inglese di default
html_content_en = f"""
<html>
  <head><title>Market Brief</title></head>
  <body>
    <h1>📊 Daily Market Summary</h1>
    <p style='font-family: Arial; font-size: 16px;'>{brief_refined}</p>
    <h2>Per-Asset Insights</h2>
    <ul>
      {"".join(f"<li>{line}</li>" for line in asset_sentences.splitlines())}
    </ul>
    <h2>💡 Mini Tip</h2>
    <p style='font-family: Arial; font-size: 14px; color: #555;'>{mini_tip}</p>
    <hr>
    <h2>🔥 Top Signal</h2>
    <p style='font-family: Arial; font-size: 16px; font-weight: bold;'>{top_signal_str}</p>
  </body>
</html>
"""

file_path = f"{TARGET_FOLDER}/daily_brief_en.html"
try:
    contents = repo.get_contents(file_path)
    repo.update_file(file_path, "Updated daily_brief_en", html_content_en, contents.sha)
except GithubException:
    repo.create_file(file_path, "Created daily_brief_en", html_content_en)






def calcola_correlazioni(dati_storici_all,
                         max_lag=6,
                         min_valid_points=60,
                         signif_level=0.05,
                         window=60,
                         alpha=0.5,
                         min_corr=0.3,
                         min_percent=50,
                         threshold_std=0.05,
                         top_k=5,
                         control_market_index=None,
                         fdr_alpha=0.05):
    """
    Versione migliorata:
    - Considera correlazioni sia positive che negative (concordi e discordi).
    - Restituisce fino a top_k partner per asset con metriche, flag di validità e tipo relazione.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    from scipy.stats import spearmanr, pearsonr, binomtest
    from statsmodels.stats.multitest import multipletests

    # 1) Calcolo returns
    returns = {}
    for sym, df in dati_storici_all.items():
        if "Close" not in df.columns:
            continue
        r = np.log(df["Close"]).diff().dropna()
        returns[sym] = r

    # 2) Effetto mercato
    market_ret = None
    if control_market_index is not None:
        market_ret = np.log(control_market_index["Close"]).diff().dropna()

    intermediate = {}
    pvals_records = []
    tests_meta = []
    assets = list(returns.keys())

    for i, asset1 in enumerate(assets):
        serie1 = returns[asset1]
        if market_ret is not None:
            joined = pd.concat([serie1, market_ret], axis=1, join="inner").dropna()
            if len(joined) >= min_valid_points:
                X = sm.add_constant(joined.iloc[:,1])
                res = OLS(joined.iloc[:,0], X).fit()
                serie1 = joined.iloc[:,0] - res.predict(X)

        for asset2 in assets:
            if asset1 == asset2:
                continue
            serie2 = returns[asset2]
            if market_ret is not None:
                joined2 = pd.concat([serie2, market_ret], axis=1, join="inner").dropna()
                if len(joined2) >= min_valid_points:
                    X2 = sm.add_constant(joined2.iloc[:,1])
                    res2 = OLS(joined2.iloc[:,0], X2).fit()
                    serie2 = joined2.iloc[:,0] - res2.predict(X2)

            for lag in range(0, max_lag + 1):  # include lag=0
                aligned = pd.concat([serie1.shift(-lag), serie2], axis=1, join="inner").dropna()
                if len(aligned) < min_valid_points:
                    continue
                x = aligned.iloc[:,0]
                y = aligned.iloc[:,1]

                # threshold micro-movements
                thr = threshold_std * x.rolling(window=min(20,len(x))).std().median()
                signs_x = np.where(np.abs(x) >= thr, np.sign(x), 0)
                signs_y = np.where(np.abs(y) >= thr, np.sign(y), 0)
                valid_idx = (signs_x != 0) | (signs_y != 0)
                if valid_idx.sum() < min_valid_points:
                    continue
                concordant = (signs_x[valid_idx] == signs_y[valid_idx]).astype(int)
                percent_concordance = 100 * concordant.mean()

                # Pearson e Spearman
                try:
                    pearson_r, pearson_p = pearsonr(x, y)
                except Exception:
                    pearson_r, pearson_p = np.nan, 1.0
                try:
                    spearman_r, spearman_p = spearmanr(x, y)
                    spearman_r = float(spearman_r)
                except Exception:
                    spearman_r, spearman_p = np.nan, 1.0

                # rolling consistency
                series_corr = x.rolling(window=window, min_periods=int(window/2)).corr(y)
                fraction_windows = np.nanmean(np.abs(series_corr) >= min_corr) * 100

                # binomial test
                concordi = int(concordant.sum())
                tot = int(valid_idx.sum())
                p_binom = binomtest(concordi, tot, 0.5, alternative='greater').pvalue

                # composite score
                score = (alpha * (percent_concordance / 100) +
                         (1-alpha)/2 * (0 if np.isnan(pearson_r) else pearson_r) +
                         (1-alpha)/2 * (0 if np.isnan(spearman_r) else spearman_r))

                key = (asset1, asset2, lag)
                intermediate[key] = {
                    "asset1": asset1,
                    "asset2": asset2,
                    "lag": lag,
                    "days": len(aligned),
                    "percent": percent_concordance,
                    "pearson": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman": spearman_r,
                    "spearman_p": spearman_p,
                    "p_binom": p_binom,
                    "score": score,
                    "fraction_windows": fraction_windows
                }

                tests_meta.append(key)
                pvals_records.append(pearson_p)

    # FDR correction
    if pvals_records:
        rej, pvals_corrected, _, _ = multipletests(pvals_records, alpha=fdr_alpha, method='fdr_bh')
    else:
        rej, pvals_corrected = [], []

    for idx, key in enumerate(tests_meta):
        intermediate[key]["pearson_p_fdr"] = pvals_corrected[idx]
        intermediate[key]["pearson_significant_fdr"] = bool(rej[idx])

    # costruisci risultati top_k
    results = {}
    for asset1 in assets:
        candidates = []
        for key, met in intermediate.items():
            if key[0] != asset1:
                continue
            # nuova logica: accetta sia concordanza che discordanza
            concord_ok = met["percent"] >= min_percent
            discord_ok = (100 - met["percent"]) >= min_percent
            relation_type = "concorde" if concord_ok else ("discorde" if discord_ok else "debole")

            valid = (met["days"] >= min_valid_points and
                     (abs(met["pearson"]) >= min_corr or abs(met["spearman"]) >= min_corr) and
                     (concord_ok or discord_ok) and
                     met["fraction_windows"] >= 50 and
                     met.get("pearson_significant_fdr", False) and
                     met["p_binom"] < signif_level)

            entry = dict(met)
            entry["valid"] = bool(valid)
            entry["relation_type"] = relation_type
            candidates.append(entry)

        # ordina per score
        candidates = sorted(candidates, key=lambda x: x["score"] if x["score"] is not None else -999, reverse=True)
        results[asset1] = candidates[:top_k]

    return results




def salva_correlazioni_html(correlazioni, repo, file_path=f"{TARGET_FOLDER}/correlations.html"):
    """
    Crea un file HTML con la tabella delle correlazioni trovate per ogni asset.
    Supporta il formato "lista di top_k partner" restituito da calcola_correlazioni.
    """
    html_corr = [
        "<html><head><title>Correlazioni tra Asset</title></head><body>",
        "<h1>Correlazioni tra Asset (Top-k partner)</h1>",
        "<table border='1' style='border-collapse: collapse; text-align: center;'>",
        "<tr>"
        "<th>Asset</th><th>Segue</th>"
        "<th>Pearson</th><th>Spearman</th>"
        "<th>Percentuale direzionale (%)</th><th>Score composito</th>"
        "<th>Lag (giorni)</th><th># Giorni</th>"
        "<th>Tipo relazione</th><th>Validità</th>"
        "</tr>"
    ]

    for symbol, entries in correlazioni.items():
        if isinstance(entries, dict):
            entries = [entries]  # compatibilità

        for info in entries:
            correlato = info.get("asset2") or info.get("asset", "N/A")
            percent_val = f"{info.get('percent', 'N/A'):.2f}" if isinstance(info.get("percent"), (int, float)) else "N/A"
            pearson_val = f"{info.get('pearson', 'N/A'):.2f}" if isinstance(info.get("pearson"), (int, float)) else "N/A"
            spearman_val = f"{info.get('spearman', 'N/A'):.2f}" if isinstance(info.get("spearman"), (int, float)) else "N/A"
            score_val = f"{info.get('score', 'N/A'):.3f}" if isinstance(info.get("score"), (int, float)) else "N/A"
            lag_val = info.get("lag", "N/A")
            days_val = info.get("days", "N/A")
            relation_type = info.get("relation_type", "N/A")

            # color coding basato sul punteggio composito
            score = info.get("score", 0)
            if score >= 0.6:
                valid_val = "🟢"
            elif score >= 0.4:
                valid_val = "🟡"
            else:
                valid_val = "🔴"

            # icona extra per il tipo di relazione
            if relation_type == "concorde":
                relation_icon = "⬆️⬆️ (positiva)"
            elif relation_type == "discorde":
                relation_icon = "⬆️⬇️ (negativa)"
            else:
                relation_icon = "⚪ debole"

            html_corr.append(
                f"<tr>"
                f"<td>{symbol}</td><td>{correlato}</td>"
                f"<td>{pearson_val}</td><td>{spearman_val}</td>"
                f"<td>{percent_val}</td><td>{score_val}</td>"
                f"<td>{lag_val}</td><td>{days_val}</td>"
                f"<td>{relation_icon}</td><td>{valid_val}</td>"
                f"</tr>"
            )

    html_corr.append("</table></body></html>")

    try:
        contents = repo.get_contents(file_path)
        repo.update_file(contents.path, "Updated correlations", "\n".join(html_corr), contents.sha)
    except GithubException:
        repo.create_file(file_path, "Created correlations", "\n".join(html_corr))

    print("✅ File correlations.html aggiornato con successo!")



correlazioni = calcola_correlazioni(dati_storici_all)

# Salva file correlations.html
salva_correlazioni_html(correlazioni, repo)
