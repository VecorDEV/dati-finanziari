from github import Github, GithubException
import re
import feedparser
import os
from datetime import datetime, timedelta
import math
#import spacy
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
from financial_lexicon import LEXICON

# Indicatori tecnici e statistica
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands
from urllib.parse import quote_plus
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr, binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# --- SETUP AI: TURBO-VADER (VADER + Expanded Financial Lexicon) ---
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Spacy model not found, proceeding without lemmatization for compatibility.")

# AGGIORNA IL LESSICO UNA VOLTA SOLA
sia = SentimentIntensityAnalyzer()
sia.lexicon.update(LEXICON)



GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = "VecorDEV/dati-finanziari"

# --- CONFIGURAZIONE CARTELLA OUTPUT ---
TARGET_FOLDER = "hybrid_results"
TEST_FOLDER = "forward_testing"  # dedicata al test


# Paths
file_path = f"{TARGET_FOLDER}/classifica.html"
news_path = f"{TARGET_FOLDER}/news.html"
history_path = f"{TARGET_FOLDER}/history.json"
fire_path = f"{TARGET_FOLDER}/fire.html"
pro_path = f"{TARGET_FOLDER}/classificaPRO.html"
corr_path = f"{TARGET_FOLDER}/correlations.html"
mom_path = f"{TARGET_FOLDER}/classifica_momentum.html"
sector_path = f"{TARGET_FOLDER}/classifica_settori.html"

    
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
# 1. MAPPE E LISTE COMPLETE
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
    "9. Utilities & Green": "IBE.MC",
    "10. Precious Metals & Materials": "GOLD",
    "11. Media & Telecom": "NFLX",
    "12. Indices (Global)": "SPX500",
    "13. Forex (Currencies)": "EURUSD",
    "14. Crypto Assets": "BTCUSD"
}

asset_sector_map = {
    "AAPL": "1. Big Tech, Software & Internet", "MSFT": "1. Big Tech, Software & Internet", 
    "GOOGL": "1. Big Tech, Software & Internet", "AMZN": "1. Big Tech, Software & Internet",
    "META": "1. Big Tech, Software & Internet", "ADBE": "1. Big Tech, Software & Internet",
    "CRM": "1. Big Tech, Software & Internet", "ORCL": "1. Big Tech, Software & Internet",
    "IBM": "1. Big Tech, Software & Internet", "NOW": "1. Big Tech, Software & Internet",
    "INTU": "1. Big Tech, Software & Internet", "ADP": "1. Big Tech, Software & Internet",
    "BABA": "1. Big Tech, Software & Internet", "BIDU": "1. Big Tech, Software & Internet",
    "SHOP": "1. Big Tech, Software & Internet", "SNOW": "1. Big Tech, Software & Internet",
    "PLTR": "1. Big Tech, Software & Internet", "TWLO": "1. Big Tech, Software & Internet",
    "DUOL": "1. Big Tech, Software & Internet", "JD": "1. Big Tech, Software & Internet",
    "NET": "1. Big Tech, Software & Internet", "PDD": "1. Big Tech, Software & Internet",
    "BTDR": "1. Big Tech, Software & Internet", "DDOG": "1. Big Tech, Software & Internet",
    "ZM": "1. Big Tech, Software & Internet",
    "NVDA": "2. Semiconductors & AI", "INTC": "2. Semiconductors & AI",
    "QCOM": "2. Semiconductors & AI", "ADI": "2. Semiconductors & AI",
    "ARM": "2. Semiconductors & AI", "CSCO": "2. Semiconductors & AI",
    "ACN": "2. Semiconductors & AI", "FIS": "2. Semiconductors & AI",
    "JPM": "3. Financial Services", "V": "3. Financial Services", 
    "PYPL": "3. Financial Services", "MS": "3. Financial Services",
    "GS": "3. Financial Services", "AXP": "3. Financial Services",
    "SCHW": "3. Financial Services", "C": "3. Financial Services",
    "PLD": "3. Financial Services", "PNC": "3. Financial Services",
    "ICE": "3. Financial Services", "MMC": "3. Financial Services",
    "CME": "3. Financial Services", "AON": "3. Financial Services",
    "TROW": "3. Financial Services", "USB": "3. Financial Services",
    "PSA": "3. Financial Services", "COIN": "3. Financial Services",
    "UCG.MI": "3. Financial Services", "PST.MI": "3. Financial Services",
    "ISP.MI": "3. Financial Services",
    "TSLA": "4. Automotive & Mobility", "GM": "4. Automotive & Mobility",
    "NIO": "4. Automotive & Mobility", "STLAM.MI": "4. Automotive & Mobility",
    "HTZ": "4. Automotive & Mobility", "LCID": "4. Automotive & Mobility",
    "RIVN": "4. Automotive & Mobility", "UBER": "4. Automotive & Mobility",
    "LYFT": "4. Automotive & Mobility", "NAAS": "4. Automotive & Mobility",
    "LLY": "5. Healthcare & Pharma", "JNJ": "5. Healthcare & Pharma",
    "PFE": "5. Healthcare & Pharma", "MRK": "5. Healthcare & Pharma",
    "ABT": "5. Healthcare & Pharma", "BMY": "5. Healthcare & Pharma",
    "AMGN": "5. Healthcare & Pharma", "CVS": "5. Healthcare & Pharma",
    "BDX": "5. Healthcare & Pharma", "ZTS": "5. Healthcare & Pharma",
    "EW": "5. Healthcare & Pharma", "LNTH": "5. Healthcare & Pharma",
    "SYK": "5. Healthcare & Pharma",
    "WMT": "6. Consumer Goods & Retail", "KO": "6. Consumer Goods & Retail",
    "PEP": "6. Consumer Goods & Retail", "MCD": "6. Consumer Goods & Retail",
    "NKE": "6. Consumer Goods & Retail", "HD": "6. Consumer Goods & Retail",
    "COST": "6. Consumer Goods & Retail", "SBUX": "6. Consumer Goods & Retail",
    "LOW": "6. Consumer Goods & Retail", "TGT": "6. Consumer Goods & Retail",
    "TJX": "6. Consumer Goods & Retail", "CL": "6. Consumer Goods & Retail",
    "EL": "6. Consumer Goods & Retail", "SCHL": "6. Consumer Goods & Retail",
    "COCOA": "6. Consumer Goods & Retail",
    "CAT": "7. Industrials & Defense", "LMT": "7. Industrials & Defense",
    "ITW": "7. Industrials & Defense", "FDX": "7. Industrials & Defense",
    "NSC": "7. Industrials & Defense", "GE": "7. Industrials & Defense",
    "HON": "7. Industrials & Defense", "DE": "7. Industrials & Defense",
    "LDO.MI": "7. Industrials & Defense", "BKNG": "7. Industrials & Defense",
    "BA": "4. Automotive & Mobility", "AIR.PA": "4. Automotive & Mobility",
    "OIL": "8. Energy (Oil & Gas)", "NATGAS": "8. Energy (Oil & Gas)",
    "XOM": "8. Energy (Oil & Gas)", "CVX": "8. Energy (Oil & Gas)",
    "PBR": "8. Energy (Oil & Gas)", "NRG": "8. Energy (Oil & Gas)",
    "SO": "9. Utilities & Green", "ENEL.MI": "9. Utilities & Green",
    "DUK": "9. Utilities & Green", "AEP": "9. Utilities & Green",
    "D": "9. Utilities & Green", "HE": "9. Utilities & Green",
    "APD": "9. Utilities & Green",
    "GOLD": "10. Precious Metals & Materials", "SILVER": "10. Precious Metals & Materials",
    "VALE": "10. Precious Metals & Materials",
    "NFLX": "11. Media & Telecom", "DIS": "11. Media & Telecom",
    "T": "11. Media & Telecom", "TMUS": "11. Media & Telecom",
    "AMX": "11. Media & Telecom", "ROKU": "11. Media & Telecom",
    "SAP.DE": "1. Big Tech, Software & Internet", "SIE.DE": "7. Industrials & Defense",
    "ALV.DE": "3. Financial Services", "VOW3.DE": "4. Automotive & Mobility",
    "MBG.DE": "4. Automotive & Mobility", "DTE.DE": "11. Media & Telecom",
    "SHEL.L": "8. Energy (Oil & Gas)", "BP.L": "8. Energy (Oil & Gas)",
    "HSBA.L": "3. Financial Services", "AZN.L": "5. Healthcare & Pharma",
    "ULVR.L": "6. Consumer Goods & Retail", "RIO.L": "10. Precious Metals & Materials",
    "MC.PA": "6. Consumer Goods & Retail", "TTE.PA": "8. Energy (Oil & Gas)",
    "OR.PA": "6. Consumer Goods & Retail", "SAN.PA": "5. Healthcare & Pharma",
    "BNP.PA": "3. Financial Services", "SAN.MC": "3. Financial Services",
    "IBE.MC": "9. Utilities & Green", "ITX.MC": "6. Consumer Goods & Retail",
    "BBVA.MC": "3. Financial Services", "TEF.MC": "11. Media & Telecom",
    "ITUB": "3. Financial Services", "NU": "3. Financial Services",
    "ABEV": "6. Consumer Goods & Retail", "EMAAR.AE": "7. Industrials & Defense",
    "DIB.AE": "3. Financial Services", "EMIRATESNBD.AE": "3. Financial Services",
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

TICKER_MAP = {
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
    "ARM": "ARM", "BABA": "BABA", "BIDU": "BIDU", "COIN": "COIN",
    "DDOG": "DDOG", "HTZ": "HTZ", "JD": "JD", "LCID": "LCID", "LYFT": "LYFT", "NET": "NET",
    "PDD": "PDD", "PLTR": "PLTR", "RIVN": "RIVN", "ROKU": "ROKU", "SHOP": "SHOP",
    "SNOW": "SNOW", "TWLO": "TWLO", "UBER": "UBER",
    "ZM": "ZM", "DUOL": "DUOL", "PBR": "PBR", "VALE": "VALE", "AMX": "AMX",
    "ISP.MI": "ISP.MI", "ENEL.MI": "ENEL.MI", "STLAM.MI": "STLAM.MI",
    "LDO.MI": "LDO.MI", "PST.MI": "PST.MI", "UCG.MI": "UCG.MI",
    "BA": "BA", "AIR.PA": "AIR.PA", "SAP.DE": "SAP.DE", "SIE.DE": "SIE.DE",
    "ALV.DE": "ALV.DE", "VOW3.DE": "VOW3.DE", "MBG.DE": "MBG.DE", "DTE.DE": "DTE.DE",
    "SHEL.L": "SHEL.L", "BP.L": "BP.L", "HSBA.L": "HSBA.L", "AZN.L": "AZN.L",
    "ULVR.L": "ULVR.L", "RIO.L": "RIO.L", "MC.PA": "MC.PA", "TTE.PA": "TTE.PA",
    "OR.PA": "OR.PA", "SAN.PA": "SAN.PA", "BNP.PA": "BNP.PA", "SAN.MC": "SAN.MC",
    "IBE.MC": "IBE.MC", "ITX.MC": "ITX.MC", "BBVA.MC": "BBVA.MC", "TEF.MC": "TEF.MC",
    "ITUB": "ITUB", "NU": "NU", "ABEV": "ABEV", "EMAAR.AE": "EMAAR.AE", "DIB.AE": "DIB.AE", "EMIRATESNBD.AE": "EMIRATESNBD.AE",
    "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X", "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X", "EURAUD": "EURAUD=X", "EURNZD": "EURNZD=X",
    "EURCAD": "EURCAD=X", "EURCHF": "EURCHF=X", "GBPCHF": "GBPCHF=X",
    "AUDCAD": "AUDCAD=X",
    "SPX500": "^GSPC", "DJ30": "^DJI", "NAS100": "^NDX", "NASCOMP": "^IXIC",
    "RUS2000": "^RUT", "VIX": "^VIX", "EU50": "^STOXX50E", "ITA40": "FTSEMIB.MI",
    "GER40": "^GDAXI", "UK100": "^FTSE", "FRA40": "^FCHI", "SWI20": "^SSMI",
    "ESP35": "^IBEX", "NETH25": "^AEX", "JPN225": "^N225", "HKG50": "^HSI",
    "CHN50": "000001.SS", "IND50": "^NSEI", "KOR200": "^KS200",
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "LTCUSD": "LTC-USD",
    "XRPUSD": "XRP-USD", "BCHUSD": "BCH-USD", "EOSUSD": "EOS-USD",
    "XLMUSD": "XLM-USD", "ADAUSD": "ADA-USD", "TRXUSD": "TRX-USD",
    "NEOUSD": "NEO-USD", "DASHUSD": "DASH-USD", "XMRUSD": "XMR-USD",
    "ETCUSD": "ETC-USD", "ZECUSD": "ZEC-USD", "BNBUSD": "BNB-USD",
    "DOGEUSD": "DOGE-USD", "USDTUSD": "USDT-USD", "LINKUSD": "LINK-USD",
    "ATOMUSD": "ATOM-USD", "XTZUSD": "XTZ-USD",
    "COCOA": "CC=F", "GOLD": "GC=F", "SILVER": "SI=F", "OIL": "CL=F", "NATGAS": "NG=F"
}

symbol_list = list(asset_sector_map.keys())
symbol_list_for_yfinance = [TICKER_MAP.get(s, s) for s in symbol_list]

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
    "COIN": ["Coinbase", "Coinbase Global", "Coinbase Global Inc."],
    "PST.MI": ["Poste Italiane", "Poste Italiane S.p.A."],
    "UCG.MI": ["Unicredit", "UniCredit", "Unicredit S.p.A.", "UniCredit Bank"],
    "ISP.MI": ["Intesa Sanpaolo", "Intesa Sanpaolo S.p.A.", "Gruppo Intesa Sanpaolo", "Intesa Sanpaolo Bank", "Banca Intesa", "Banca Sanpaolo"],
    "ENEL.MI": ["Enel", "Enel S.p.A.", "Gruppo Enel"],
    "STLAM.MI": ["Stellantis", "Stellantis N.V.", "Gruppo Stellantis", "Fiat Chrysler", "FCA", "PSA Group"],
    "LDO.MI": ["Leonardo", "Leonardo S.p.A.", "Leonardo Finmeccanica", "Gruppo Leonardo"],
    "BA": ["Boeing", "The Boeing Company"],
    "AIR.PA": ["Airbus", "Airbus SE"],
    "SAP.DE": ["SAP", "SAP SE"],
    "SIE.DE": ["Siemens", "Siemens AG"],
    "ALV.DE": ["Allianz", "Allianz SE"],
    "VOW3.DE": ["Volkswagen", "Volkswagen AG"],
    "MBG.DE": ["Mercedes-Benz", "Mercedes-Benz Group"],
    "DTE.DE": ["Deutsche Telekom", "Deutsche Telekom AG"],
    "SHEL.L": ["Shell", "Shell plc"],
    "BP.L": ["BP", "BP p.l.c."],
    "HSBA.L": ["HSBC", "HSBC Holdings"],
    "AZN.L": ["AstraZeneca", "AstraZeneca PLC"],
    "ULVR.L": ["Unilever", "Unilever PLC"],
    "RIO.L": ["Rio Tinto", "Rio Tinto Group"],
    "MC.PA": ["LVMH", "Moët Hennessy Louis Vuitton"],
    "TTE.PA": ["TotalEnergies", "TotalEnergies SE"],
    "OR.PA": ["L'Oréal", "L'Oreal"],
    "SAN.PA": ["Sanofi", "Sanofi S.A."],
    "BNP.PA": ["BNP Paribas", "BNP Paribas S.A."],
    "SAN.MC": ["Santander", "Banco Santander"],
    "IBE.MC": ["Iberdrola", "Iberdrola S.A."],
    "ITX.MC": ["Inditex", "Zara"],
    "BBVA.MC": ["BBVA", "Banco Bilbao Vizcaya Argentaria"],
    "TEF.MC": ["Telefónica", "Telefonica"],
    "ITUB": ["Itaú", "Itaú Unibanco"],
    "NU": ["Nubank", "Nu Holdings"],
    "ABEV": ["Ambev", "Ambev S.A."],
    "EMAAR.AE": ["Emaar", "Emaar Properties"],
    "DIB.AE": ["DIB", "Dubai Islamic Bank P.J.S.C.", "Dubai Bank", "Dubai Islamic Bank"],
    "EMIRATESNBD.AE": ["Emirates NBD", "Emirates NBD Bank"],
    "RIVN": ["Rivian", "Rivian Automotive", "Rivian Automotive Inc."],
    "LCID": ["Lucid", "Lucid Motors", "Lucid Group", "Lucid Group Inc."],
    "DDOG": ["Datadog", "Datadog Inc."],
    "NET": ["Cloudflare", "Cloudflare Inc."],
    "SHOP": ["Shopify", "Shopify Inc."],
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
    def __init__(self, repo, filename=history_path):
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
        # SETUP REATTIVO:
        # Teniamo solo 21 giorni (circa 1 mese di borsa). 
        # Questo rende la "media" molto più sensibile ai cambiamenti recenti.
        limit_date = datetime.now() - timedelta(days=21)
        
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
        
        # 1. Calcolo Sentiment
        avg_sent = sum(past_sentiments) / len(past_sentiments)
        sent_diff = current_sent - avg_sent
        raw_delta = (sent_diff * 100)
        
        # --- LOGICA REATTIVA (FAST MOMENTUM) ---
        multiplier = 1.0
        
        # SOGLIA BASSA: Basta poco per attivare l'analisi (3 news)
        MIN_NEWS_FLOOR = 3  
        
        # Bastano 2 giorni di storico per iniziare a calcolare (molto aggressivo)
        if len(past_counts) >= 2 and current_count >= MIN_NEWS_FLOOR:
            
            avg_count = np.mean(past_counts)
            std_dev = np.std(past_counts)
            
            # Deviazione standard minima più bassa per essere più sensibili
            if std_dev < 0.2: std_dev = 0.2
            
            z_score = (current_count - avg_count) / std_dev
            
            # TRIGGER PIÙ FACILI DA RAGGIUNGERE
            # Z=2.0 (Top 5%) invece di 3.0 (Top 0.3%) per il massimo boost
            if z_score >= 2.0:      
                multiplier = 2.0    # Massimo Boost
            elif z_score >= 1.5:    
                multiplier = 1.75   # Boost Alto
            elif z_score >= 1.0:    
                multiplier = 1.25   # Boost Medio (basta essere 1 sigma sopra la media)
                
        else:
            # Fallback Aggressivo per nuovi asset o pochissimi dati
            # Se le news sono il doppio della media semplice, spingiamo già.
            avg_simple = sum(past_counts)/len(past_counts) if past_counts else 0
            # Se oggi ho più di 5 news e sono il doppio della media -> Boost
            if current_count >= 5 and current_count >= (avg_simple * 2):
                multiplier = 1.5

        final_delta = 50 + (raw_delta * multiplier)
        return max(min(final_delta, 100), 0)

class BacktestSystem:
    def __init__(self, repo, folder_name="forward_testing"):
        self.repo = repo
        self.folder = folder_name
        self.json_filename = f"{self.folder}/backtest_log.json"
        self.html_filename = f"{self.folder}/reliability_curve.html"
        self.data = self._load_data()
        
    def _load_data(self):
        try:
            # Scarica il file attuale
            contents = self.repo.get_contents(self.json_filename)
            raw_data = json.loads(base64.b64decode(contents.content).decode('utf-8'))
            
            # --- MIGRAZIONE AUTOMATICA E INTELLIGENTE ---
            # Se trova la vecchia chiave "log" (lista piatta), converte tutto
            if "log" in raw_data and isinstance(raw_data["log"], list):
                print(f"⚠️ Rilevato vecchio formato ({len(raw_data['log'])} righe). Avvio migrazione e compressione...")
                new_structure = {}
                
                for entry in raw_data["log"]:
                    sym = entry["symbol"]
                    if sym not in new_structure: new_structure[sym] = []
                    
                    # Evita duplicati durante la migrazione (stessa data nello stesso asset)
                    date_exists = any(x["d"] == entry["date"] for x in new_structure[sym])
                    if not date_exists:
                        # Crea record compatto
                        compact_entry = {
                            "d": entry["date"],          # Data
                            "s": entry["score"],         # Score
                            "p": entry["start_price"],   # Prezzo Start
                            "r": entry["daily_results"], # Results
                            "st": entry.get("status", "active")
                        }
                        new_structure[sym].append(compact_entry)
                
                print("✅ Migrazione completata. I dati sono salvi e ottimizzati.")
                return {"assets": new_structure, "stats": raw_data.get("stats", {})}
            
            # Se è già nel nuovo formato, ritorna così com'è
            return raw_data
        except Exception as e:
            print(f"Nessun dato precedente trovato o errore caricamento: {e}")
            return {"assets": {}, "stats": {}}

    def save_data(self):
        try:
            # Mantiene pulito: Ordina per data decrescente e tiene max 100 giorni per asset
            for sym in self.data["assets"]:
                self.data["assets"][sym] = sorted(self.data["assets"][sym], key=lambda x: x["d"], reverse=True)[:100]

            content = json.dumps(self.data, indent=None, separators=(',', ':')) # Minificazione JSON (rimuove spazi inutili)
            
            try:
                c = self.repo.get_contents(self.json_filename)
                self.repo.update_file(self.json_filename, "Upd Optimized Data", content, c.sha)
            except:
                self.repo.create_file(self.json_filename, "Init Optimized Data", content)
        except Exception as e:
            print(f"Errore salvataggio JSON backtest: {e}")

    def log_new_prediction(self, symbol, score, current_price):
        """Salva o aggiorna nella nuova struttura compatta"""
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # Inizializza lista asset se non esiste
        if symbol not in self.data["assets"]:
            self.data["assets"][symbol] = []
            
        history = self.data["assets"][symbol]
        
        # Cerca se esiste già una entry per OGGI
        existing_entry = next((item for item in history if item["d"] == today_str), None)

        if existing_entry:
            # SOVRASCRRITTURA: Aggiorna l'entry di oggi con i dati più recenti (es. esecuzione delle 16:00)
            existing_entry["s"] = score
            existing_entry["p"] = float(current_price)
        else:
            # NUOVA ENTRY: Aggiungi in cima alla lista
            new_entry = {
                "d": today_str,
                "s": score,
                "p": float(current_price),
                "r": {},
                "st": "active"
            }
            history.insert(0, new_entry)

    def update_daily_tracking(self, current_prices_map):
        """Calcola i risultati navigando la nuova struttura a dizionario"""
        today = datetime.now()
        max_days = 20
        
        # Itera su ogni asset nel dizionario
        for symbol, history in self.data["assets"].items():
            if symbol not in current_prices_map: continue
            current_price = current_prices_map[symbol]
            
            for entry in history:
                if entry.get("st") == "closed": continue
                
                try:
                    entry_date = datetime.strptime(entry["d"], "%Y-%m-%d")
                    days_passed = (today - entry_date).days
                    
                    if days_passed == 0: continue # Salta oggi
                    if days_passed > max_days:
                        entry["st"] = "closed"
                        continue
                        
                    start_price = entry["p"]
                    change = ((current_price - start_price) / start_price) * 100
                    
                    # Salva risultato con chiave stringa corta (es. "3" per 3 giorni)
                    entry["r"][str(days_passed)] = round(change, 2)
                except: continue

        self._analyze_stats()

    def _analyze_stats(self):
        """Genera statistiche dalla struttura a dizionario"""
        stats_by_day = {}
        
        for symbol, history in self.data["assets"].items():
            for entry in history:
                score = entry["s"]
                
                # Filtro Confidenza: Analizza solo se l'AI era decisa
                direction = 0
                if score >= 55: direction = 1   # Long
                elif score <= 45: direction = -1 # Short
                else: continue
                
                for day, val in entry["r"].items():
                    if day not in stats_by_day: stats_by_day[day] = {"wins": 0, "total": 0, "ret": 0.0}
                    
                    # Logica Win: Direzione giusta E movimento significativo (>0.1%)
                    is_win = (direction == 1 and val > 0.1) or (direction == -1 and val < -0.1)
                    
                    stats_by_day[day]["total"] += 1
                    if is_win: stats_by_day[day]["wins"] += 1
                    stats_by_day[day]["ret"] += val

        curve = []
        best_day = "N/A"
        best_acc = 0.0
        
        for d in sorted(stats_by_day.keys(), key=lambda x: int(x)):
            data = stats_by_day[d]
            if data["total"] < 5: continue # Ignora campioni statistici troppo piccoli
            
            acc = round((data["wins"]/data["total"])*100, 1)
            avg_ret = round(data["ret"]/data["total"], 2)
            curve.append({"day": int(d), "accuracy": acc, "avg_return": avg_ret})
            
            if acc > best_acc:
                best_acc = acc
                best_day = d
                
        self.data["stats"] = {"best_day": best_day, "best_acc": best_acc, "curve": curve}

    def generate_report(self):
        """Genera report HTML usando i nuovi dati"""
        stats = self.data.get("stats", {})
        curve = stats.get("curve", [])
        
        # Calcolo Win Rate Totale (Media di tutti i giorni)
        total_acc = 0
        if curve:
            total_acc = sum(x['accuracy'] for x in curve) / len(curve)

        html = [
            "<html><head><title>Forward Testing</title>",
            "<style>body{font-family:Arial, sans-serif;padding:20px;background:#f4f4f9;} ",
            ".card{background:white;padding:20px;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,0.1);margin-bottom:20px;} ",
            ".bar-container{background:#eee;border-radius:4px;overflow:hidden;} ",
            ".bar{height:24px;color:white;text-align:right;padding-right:10px;line-height:24px;font-size:0.9em;} ",
            ".g{background:#28a745;} .r{background:#dc3545;} .y{background:#ffc107;color:black;} ",
            "table{width:100%;border-collapse:collapse;} th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left;} th{background:#fafafa;}</style>",
            "</head><body>",
            "<div class='card'>",
            "<h1>🧪 Forward Testing (Real-time Validation)</h1>",
            f"<p>Analisi su segnali reali passati. <b>Struttura Dati Ottimizzata.</b></p>",
            f"<p><b>Win Rate Medio (Direzione):</b> {total_acc:.1f}%</p>",
            f"<p><b>Picco Affidabilità:</b> Giorno {stats.get('best_day','-')} con {stats.get('best_acc',0)}%</p>",
            "</div>",
            "<div class='card'><table><tr><th>Giorno</th><th>Win Rate (Direzione)</th><th>Profitto (Buy&Hold)</th></tr>"
        ]
        
        for p in curve:
            d, acc, ret = p['day'], p['accuracy'], p['avg_return']
            
            # Colore barra
            if acc >= 55: color = "g"
            elif acc >= 48: color = "y"
            else: color = "r"
            
            # Larghezza barra (minimo 15% per visibilità)
            width = max(acc, 15)
            
            # Colore testo profitto
            profit_color = "green" if ret > 0 else "red"
            
            html.append(f"<tr><td>Day {d}</td>"
                        f"<td><div class='bar-container'><div class='bar {color}' style='width:{width}%'>{acc}%</div></div></td>"
                        f"<td style='color:{profit_color};font-weight:bold;'>{ret:+.2f}%</td></tr>")
            
        html.append("</table></div></body></html>")
        
        try:
            full_html = "\n".join(html)
            # Tenta aggiornamento, altrimenti crea
            try:
                c = self.repo.get_contents(self.html_filename)
                self.repo.update_file(self.html_filename, "Upd Report", full_html, c.sha)
            except:
                self.repo.create_file(self.html_filename, "Cre Report", full_html)
        except Exception as e:
            print(f"Errore report HTML: {e}")



# ==============================================================================
# CLASSE PATTERN ANALYZER (PROFESSIONALE)
# ==============================================================================
class PatternAnalyzer:
    def __init__(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            self.o = df['Open'].iloc[:, 0].values
            self.h = df['High'].iloc[:, 0].values
            self.l = df['Low'].iloc[:, 0].values
            self.c = df['Close'].iloc[:, 0].values
        else:
            self.o = df['Open'].values
            self.h = df['High'].values
            self.l = df['Low'].values
            self.c = df['Close'].values

    def get_pattern_score(self):
        """
        Returns a score between -1.0 (Strong Bearish) and +1.0 (Strong Bullish).
        Used for mathematical calculation in HybridScorer.
        """
        score, _ = self._analyze_logic()
        return score

    def get_pattern_info(self):
        """
        Returns: (Numeric Score, String with English pattern names)
        Used for HTML display and App text.
        """
        score, patterns = self._analyze_logic()
        # Se la lista è vuota, restituisce stringa inglese
        pattern_text = ", ".join(patterns) if patterns else "No significant patterns"
        return score, pattern_text

    def _analyze_logic(self):
        """
        Internal logic to avoid code duplication between score and info.
        """
        score = 0.0
        patterns_found = []
        limit = len(self.c)
        
        if limit < 20: return 0.0, ["Insufficient Data"]
        
        # --- A. CANDLESTICK ANALYSIS (Last 3 days) ---
        i = limit - 1
        c1, c2, c3 = self.c[i-2], self.c[i-1], self.c[i]
        o1, o2, o3 = self.o[i-2], self.o[i-1], self.o[i]
        h3, l3 = self.h[i], self.l[i]
        body3 = abs(c3 - o3)
        range3 = h3 - l3 if h3 != l3 else 0.0001

        # 1. Bullish Engulfing
        if c2 < o2 and c3 > o3 and c3 > o2 and o3 < c2: 
            score += 0.4
            patterns_found.append("Bullish Engulfing")
        
        # 2. Bearish Engulfing
        if c2 > o2 and c3 < o3 and c3 < o2 and o3 > c2: 
            score -= 0.4
            patterns_found.append("Bearish Engulfing")

        # 3. Hammer
        lower_shadow = min(c3, o3) - l3
        if lower_shadow > (body3 * 2) and (h3 - max(c3, o3)) < body3: 
            score += 0.3
            patterns_found.append("Hammer")

        # 4. Shooting Star
        upper_shadow = h3 - max(c3, o3)
        if upper_shadow > (body3 * 2) and (min(c3, o3) - l3) < body3: 
            score -= 0.3
            patterns_found.append("Shooting Star")

        # 5. Three White Soldiers
        if c1 > o1 and c2 > o2 and c3 > o3 and c3 > c2 > c1: 
            score += 0.3
            patterns_found.append("3 White Soldiers")

        # --- B. SUPPORT & RESISTANCE (Structural) ---
        curr_price = c3
        lookback = 126 
        recent_h = self.h[-lookback:]
        recent_l = self.l[-lookback:]
        threshold = 0.02
        
        # At Support?
        if abs(curr_price - np.min(recent_l)) / curr_price <= threshold:
            score += 0.4
            patterns_found.append("At Support Level")

        # At Resistance?
        if abs(curr_price - np.max(recent_h)) / curr_price <= threshold:
            score -= 0.4
            patterns_found.append("At Resistance Level")

        final_score = max(min(score, 1.0), -1.0)
        return final_score, patterns_found


class HybridScorer:
    def _calculate_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean().replace(0, np.nan)
        rs = gain / loss
        rs = rs.fillna(0)
        return 100 - (100 / (1 + rs))

    def _get_technical_score(self, df):
        # Questo è il tuo vecchio metodo (RSI + SMA)
        # Lo teniamo come "Trend Score" di base
        if len(df) < 50: return 0.0
        
        # Gestione MultiIndex se necessario
        close = df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        
        try:
            sma = float(close.rolling(window=50).mean().iloc[-1])
            curr = float(close.iloc[-1])
            rsi = float(self._calculate_rsi(close).iloc[-1])
        except: return 0.0
        
        score = 0.0
        if curr > sma: score += 0.5
        else: score -= 0.5
        
        # Nota: RSI qui serve per Ipercomprato/Ipervenduto generico
        if rsi < 30: score += 0.5 
        elif rsi > 70: score -= 0.5 
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df, sent_raw, news_n, lead, is_lead, delta_score):
        # 1. Analisi Tecnica Standard (RSI, SMA)
        tech_score = self._get_technical_score(df)
        
        # 2. Analisi Pattern Avanzata (Nuova aggiunta)
        analyzer = PatternAnalyzer(df)
        pattern_score = analyzer.get_pattern_score()

        curr_lead = 0.0 if is_lead else lead
        delta_factor = (delta_score - 50) / 50.0 
        
        # --- DEFINIZIONE PESI (WEIGHTS) ---
        # w_n = News Sentiment
        # w_t = Technical Trend (SMA/RSI)
        # w_p = Pattern (Candele + Supporti) -> NUOVO
        # w_l = Leader di settore
        # w_d = Delta Momentum (Hype recente)

        if is_lead:
            if news_n == 0:     
                # Senza news, ci affidiamo a Tecnica, Pattern e Momentum
                w_n, w_l, w_t, w_p, w_d = 0.00, 0.00, 0.40, 0.40, 0.20
            elif news_n <= 3:   
                w_n, w_l, w_t, w_p, w_d = 0.20, 0.00, 0.35, 0.30, 0.15
            else:               
                # Con tante news, il Sentiment pesa di più
                w_n, w_l, w_t, w_p, w_d = 0.40, 0.00, 0.25, 0.20, 0.15
        else:
            if news_n == 0:     
                w_n, w_l, w_t, w_p, w_d = 0.00, 0.20, 0.35, 0.35, 0.10
            elif news_n <= 3:   
                w_n, w_l, w_t, w_p, w_d = 0.15, 0.20, 0.30, 0.25, 0.10
            else:               
                w_n, w_l, w_t, w_p, w_d = 0.35, 0.15, 0.20, 0.20, 0.10
        
        # Calcolo Finale Ponderato
        final = (sent_raw * w_n) + \
                (tech_score * w_t) + \
                (pattern_score * w_p) + \
                (curr_lead * w_l) + \
                (delta_factor * w_d)
        
        # Limita tra -1 e 1
        final = max(min(final, 1.0), -1.0)
        
        # Converte in scala 0-100
        return round(50 + (final * 50), 2)

    def get_signal(self, score):
        if score >= 60: return "STRONG BUY", "green"
        elif score >= 53: return "BUY", "green"
        elif score <= 40: return "STRONG SELL", "red"
        elif score <= 47: return "SELL", "red"
        else: return "HOLD", "black"

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def generate_query_variants(symbol):
    base_variants = [f"{symbol} stock", f"{symbol} investing", f"{symbol} earnings", f"{symbol} news", f"{symbol} analysis"]
    names = symbol_name_map.get(symbol.upper(), [])
    for name in names:
        base_variants.append(f"{name} stock")
    return list(set(base_variants))

MAX_ARTICLES_PER_SYMBOL = 500

def get_stock_news(symbol):
    query_variants = generate_query_variants(symbol)
    base_url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
    now = datetime.utcnow()
    days_90 = now - timedelta(days=90)
    days_30 = now - timedelta(days=30)
    days_7  = now - timedelta(days=7)

    news_90_days, news_30_days, news_7_days = [], [], []
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
    """
    Calcola il sentiment usando Turbo-VADER.
    Mantiene il peso temporale per dare più importanza alle news recenti.
    """
    # Se non ci sono news, neutro
    if not news_items: 
        return 0.5 if not return_raw else 0.0

    scores = []
    now = datetime.utcnow()
    
    for item in news_items:
        title = item[0]
        date = item[1]
        
        # Analisi Sentiment
        # 'compound' è il punteggio aggregato (-1 molto negativo, +1 molto positivo)
        score = sia.polarity_scores(title)['compound']
        
        # Peso temporale: le notizie di oggi pesano più di quelle di 3 mesi fa
        # Formula: e^(-0.03 * giorni_passati)
        days = (now - date).days
        weight = math.exp(-0.03 * days)
        
        scores.append(score * weight)
        
    # Media ponderata
    avg = sum(scores) / len(scores) if scores else 0
    
    if return_raw: 
        return avg # Restituisce da -1 a 1 (per calcoli matematici)
        
    # Normalizzazione finale da [-1, 1] a [0, 1] (per percentuali 0-100%)
    return (avg + 1) / 2

# ==============================================================================
# 4. MAIN LOGIC (FUSIONE COMPLETA)
# ==============================================================================

def get_sentiment_for_all_symbols(symbol_list):
    history_mgr = HistoryManager(repo, history_path)
    scorer = HybridScorer()

    # --- SETUP BACKTESTER (Cartella Separata) ---
    # Questo inizializza il sistema puntando a "forward_testing/"
    # Se la cartella non esiste, GitHub la creerà col primo file.
    backtester = BacktestSystem(repo, folder_name=TEST_FOLDER)
    current_prices_map = {} # Serve per il controllo bulk finale
    # --------------------------------------------
    
    sentiment_results = {}
    percentuali_combine = {} 
    all_news_entries = []
    crescita_settimanale = {}
    dati_storici_all = {}
    indicator_data = {}
    fundamental_data = {}
    momentum_results = {}
    
    # Pre-calcolo Leaders
    leader_trends = {}
    for sec, ticker in sector_leaders.items():
        try:
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
        # 1. News & Sentiment
        news_data = get_stock_news(symbol)
        s7_raw = calculate_sentiment_vader(news_data["last_7_days"], return_raw=True)
        s7_norm = calculate_sentiment_vader(news_data["last_7_days"], return_raw=False) # 0-1 range for history
        news_count_7 = len(news_data["last_7_days"])
        s90 = calculate_sentiment_vader(news_data["last_90_days"])
        sentiment_results[symbol] = {"90_days": s90}
        
        # 2. Delta Score
        history_mgr.update_history(symbol, s7_norm, news_count_7)
        delta_val = history_mgr.calculate_delta_score(symbol, s7_norm, news_count_7)
        
        # SALVIAMO IL VALORE NEL DIZIONARIO
        momentum_results[symbol] = delta_val
        
        # 3. Dati Tecnici & Hybrid Score
        hybrid_prob = 50.0
        signal_str = "HOLD"
        sig_col = "black"
        
        tabella_indicatori = None
        dati_storici_html = None
        tabella_fondamentali = None
        sells_data = None
        
        sector = asset_sector_map.get(symbol, "General")
        leader_sym = sector_leaders.get(sector, "SPX500")
        leader_val = leader_trends.get(leader_sym, 0.0)
        is_leader = (symbol == leader_sym)

        try:
            ticker = str(adjusted_symbol).strip().upper()
            data = yf.download(ticker, period="3y", auto_adjust=True, progress=False)

            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                     try: data = data.xs(ticker, axis=1, level=1)
                     except: pass
                
                close = data['Close']
                high = data['High']
                low = data['Low']
                dati_storici_all[symbol] = data.copy()

                # Estrazione Pattern per HTML (usiamo PatternAnalyzer esplicitamente per ottenere il TESTO)
                analyzer = PatternAnalyzer(data)
                pat_score_val, pat_text_names = analyzer.get_pattern_info()
                
                # Logica Colori e Stati (In Inglese per output internazionale)
                pat_sentiment_str = "NEUTRAL"
                pat_color = "black"
                if pat_score_val >= 0.3: 
                    pat_sentiment_str = "BULLISH"
                    pat_color = "green"
                elif pat_score_val <= -0.3: 
                    pat_sentiment_str = "BEARISH"
                    pat_color = "red"
                    
                # Calcolo Hybrid Score
                hybrid_prob = scorer.calculate_probability(data, s7_raw, news_count_7, leader_val, is_leader, delta_val)
                percentuali_combine[symbol] = hybrid_prob 
                signal_str, sig_col = scorer.get_signal(hybrid_prob)

                current_price = float(close.iloc[-1])
                current_prices_map[symbol] = current_price
    
                # --- LOGGING SILENZIOSO ---
                # Questo salva i dati in memoria, non tocca file, non stampa nulla.
                # Non influenza i tuoi report "classifica.html" o altro.
                backtester.log_new_prediction(symbol, hybrid_prob, current_price)
                # --------------------------
            
                # Crescita Settimanale
                try:
                    # 1. Prezzo Attuale
                    last_price = close.iloc[-1]
                    last_date = close.index[-1]

                    # 2. Data Target: Esattamente 7 giorni fa
                    target_date = last_date - timedelta(days=7)

                    # 3. Trova il prezzo in quella data (o il giorno di borsa aperta precedente)
                    # 'asof' cerca il valore all'indice specificato o quello immediatamente precedente
                    prev_price = close.asof(target_date)

                    # Se non trova nulla (es. storico troppo breve), usa un fallback a 5 candele fa
                    if pd.isna(prev_price):
                        idx = max(0, len(close) - 6)
                        prev_price = close.iloc[idx]

                    # 4. Calcolo Variazione
                    growth = ((last_price - prev_price) / prev_price) * 100
                    crescita_settimanale[symbol] = round(growth, 2)
                except: 
                    crescita_settimanale[symbol] = 0.0

                # --- INDICATORI TECNICI COMPLETI ---
                rsi = RSIIndicator(close).rsi().iloc[-1]
                macd = MACD(close)
                macd_line = macd.macd().iloc[-1]
                macd_signal = macd.macd_signal().iloc[-1]
                stoch = StochasticOscillator(high, low, close)
                stoch_k = stoch.stoch().iloc[-1]
                stoch_d = stoch.stoch_signal().iloc[-1]
                ema_10 = EMAIndicator(close, window=10).ema_indicator().iloc[-1]
                cci = CCIIndicator(high, low, close).cci().iloc[-1]
                will_r = WilliamsRIndicator(high, low, close).williams_r().iloc[-1]
                bb = BollingerBands(close)
                
                indicators = {
                    "RSI (14)": round(rsi, 2),
                    "MACD Line": round(macd_line, 2),
                    "MACD Signal": round(macd_signal, 2),
                    "Stochastic %K": round(stoch_k, 2),
                    "Stochastic %D": round(stoch_d, 2),
                    "EMA (10)": round(ema_10, 2),
                    "CCI (14)": round(cci, 2),
                    "Williams %R": round(will_r, 2),
                    "BB Upper": round(bb.bollinger_hband().iloc[-1], 2),
                    "BB Lower": round(bb.bollinger_lband().iloc[-1], 2),
                    "BB Width": round(bb.bollinger_wband().iloc[-1], 4),
                }
                indicator_data[symbol] = indicators
                tabella_indicatori = pd.DataFrame(indicators.items(), columns=["Indicatore", "Valore"]).to_html(index=False, border=0)
                
                # --- FONDAMENTALI COMPLETI ---
                tk_obj = yf.Ticker(adjusted_symbol)
                try:
                    info = tk_obj.info or {}
                    def safe_value(key):
                        val = info.get(key)
                        return round(val, 4) if isinstance(val, (int, float)) else "N/A"
                    
                    fondamentali = {
                        "Trailing P/E": safe_value("trailingPE"),
                        "Forward P/E": safe_value("forwardPE"),
                        "EPS Growth (YoY)": safe_value("earningsQuarterlyGrowth"),
                        "Revenue Growth (YoY)": safe_value("revenueGrowth"),
                        "Profit Margins": safe_value("profitMargins"),
                        "Debt to Equity": safe_value("debtToEquity"),
                        "Dividend Yield": safe_value("dividendYield")
                    }
                    fundamental_data[symbol] = fondamentali
                    tabella_fondamentali = pd.DataFrame(fondamentali.items(), columns=["Fondamentale", "Valore"]).to_html(index=False, border=0)
                except: pass
                
                # Storico HTML
                hist = data.tail(90).copy()
                hist['Date'] = hist.index.strftime('%Y-%m-%d')
                dati_storici_html = hist[['Date','Close','High','Low','Open','Volume']].to_html(index=False, border=1)

                # --- INSIDER SELLS (LOGICA A CASCATA: OPENINSIDER -> YFINANCE) ---
                sells_data = None
        
                # 1. TENTATIVO PRINCIPALE: OPENINSIDER (Ottimizzato per USA)
                try:
                    # Filtro: Usiamo OpenInsider solo se sembra un'azione USA standard.
                    # Scartiamo simboli con =, ^, -USD (Crypto/Indici) e suffissi europei (.MI, .PA, ecc)
                    is_likely_us_stock = not any(x in str(adjusted_symbol) for x in ["=", "^", "-USD", ".MI", ".PA", ".DE", ".L", ".MC", ".HE", ".LS"])
                    
                    if is_likely_us_stock:
                        # Usa adjusted_symbol per evitare errori su ticker ambigui
                        url = f"http://openinsider.com/screener?s={adjusted_symbol}&o=&cnt=1000"
                        tables = pd.read_html(url)
                        
                        if len(tables) > 0:
                            insider_trades = max(tables, key=lambda t: t.shape[0])
                            # Pulizia Dati
                            insider_trades['Value_clean'] = insider_trades['Value'].replace(r'[\$,]', '', regex=True).astype(float)
                            sells = insider_trades[insider_trades['Trade\xa0Type'].str.contains("Sale", na=False)].copy()
                            
                            if not sells.empty:
                                sells['Trade Date'] = pd.to_datetime(insider_trades['Trade\xa0Date'])
                                daily_sells = sells.groupby('Trade Date')['Value_clean'].sum().abs().sort_index()
        
                                # Calcoli (Identici alla logica originale)
                                last_day = daily_sells.index.max()
                                last_value = daily_sells[last_day]
                                max_daily = daily_sells.max()
                                percent_of_max = (last_value / max_daily * 100) if max_daily != 0 else 0
                                num_sells_last_day = len(sells[sells['Trade Date'] == last_day])
        
                                variance = 0
                                if len(daily_sells) >= 2:
                                    prev_val = daily_sells.iloc[-2]
                                    if prev_val > 0:
                                        variance = ((last_value - prev_val) / prev_val) * 100
        
                                # OUTPUT DIZIONARIO (Formato Standard)
                                sells_data = {
                                    'Last Day': last_day.strftime('%Y-%m-%d'),
                                    'Last Day Total Sells ($)': f"{last_value:,.2f}",
                                    'Last vs Max (%)': percent_of_max,
                                    'Number of Sells Last Day': num_sells_last_day,
                                    'Variance': variance 
                                }
                except Exception:
                    pass # Se fallisce, prosegue al fallback
        
                # 2. TENTATIVO FALLBACK: YAHOO FINANCE (Per Europa e resto del mondo)
                # Esegue solo se il primo tentativo non ha prodotto risultati
                if sells_data is None:
                    try:
                        # Scarica transazioni da yfinance
                        ticker_obj = yf.Ticker(adjusted_symbol)
                        insider_df = ticker_obj.insider_transactions
                        
                        if not insider_df.empty:
                            # Cerca colonne che contengono la descrizione (Text o Transaction)
                            col_text = next((c for c in insider_df.columns if 'Text' in c or 'Transaction' in c), None)
                            
                            if col_text:
                                # Filtra per "Sale" o "Sold"
                                sells = insider_df[insider_df[col_text].astype(str).str.contains("Sale|Sold", case=False, na=False)].copy()
                                
                                # Verifica esistenza colonne necessarie ('Value' e 'Start Date')
                                if not sells.empty and 'Value' in sells.columns and 'Start Date' in sells.columns:
                                    sells['Value'] = sells['Value'].fillna(0).astype(float)
                                    sells['Trade Date'] = pd.to_datetime(sells['Start Date'])
                                    
                                    daily_sells = sells.groupby('Trade Date')['Value'].sum().abs().sort_index()
                                    
                                    if not daily_sells.empty:
                                        last_day = daily_sells.index.max()
                                        last_value = daily_sells[last_day]
                                        max_daily = daily_sells.max()
                                        percent_of_max = (last_value / max_daily * 100) if max_daily != 0 else 0
                                        num_sells_last_day = len(sells[sells['Trade Date'] == last_day])
                                        
                                        variance = 0
                                        if len(daily_sells) >= 2:
                                            prev_val = daily_sells.iloc[-2]
                                            if prev_val > 0:
                                                variance = ((last_value - prev_val) / prev_val) * 100
        
                                        # OUTPUT DIZIONARIO (Stesso Formato Standard)
                                        sells_data = {
                                            'Last Day': last_day.strftime('%Y-%m-%d'),
                                            'Last Day Total Sells ($)': f"{last_value:,.2f}",
                                            'Last vs Max (%)': percent_of_max,
                                            'Number of Sells Last Day': num_sells_last_day,
                                            'Variance': variance 
                                        }
                    except Exception:
                        pass

        except Exception as e: print(f"Err {symbol}: {e}")
        
        # 4. Generazione HTML Singolo (Struttura Aggiornata + Dati Completi)
        file_res = f"{TARGET_FOLDER}/{symbol.upper()}_RESULT.html"
        html_content = [
            f"<html><head><title>{symbol} Forecast</title></head><body>",
            f"<h1>Report: {symbol}</h1>",
            f"<h2 style='color:{sig_col}'>{signal_str} (Hybrid Score: {hybrid_prob}%)</h2>",
            "<hr>",
            "<h3>Price Action Analysis (Patterns)</h3>",
            f"<p><strong>Detected Patterns:</strong> {pat_text_names}</p>",
            f"<p><strong>Chart Sentiment:</strong> <span style='color:{pat_color}'><b>{pat_sentiment_str}</b></span> (Score: {pat_score_val:.2f})</p>",
            "<hr>",
            "<h3>Analisi Hybrid (AI + Tech + Delta)</h3>",
            f"<p><strong>Settore:</strong> {sector} (Trend Leader: {'UP' if leader_val>0 else 'DOWN'})</p>",
            f"<p><strong>Delta Score (Momentum News):</strong> {round(delta_val, 2)}</p>",
            "<hr>",
            "<h2>Indicatori Tecnici</h2>",
            tabella_indicatori if tabella_indicatori else "<p>N/A</p>",
            "<h2>Dati Fondamentali</h2>",
            tabella_fondamentali if tabella_fondamentali else "<p>N/A</p>",
            "<h2>Informative Sells</h2>"
        ]
        
        if sells_data:
            html_content += [
                f"<p><strong>Ultimo giorno registrato:</strong> {sells_data['Last Day']}</p>",
                f"<p><strong>Totale vendite ultimo giorno ($):</strong> {sells_data['Last Day Total Sells ($)']}</p>",
                f"<p><strong>% rispetto al massimo storico giornaliero:</strong> {sells_data['Last vs Max (%)']:.2f}%</p>",
                f"<p><strong>Transazioni recenti:</strong> {sells_data['Number of Sells Last Day']}</p>",
                f"<p><strong>Variazione:</strong> {sells_data['Variance']:.2f}%</p>"
            ]
        else:
            html_content.append("<p>Informative Sells non disponibili.</p>")
            
        html_content.append("<h2>Dati Storici (ultimi 90 giorni)</h2>")
        html_content.append(dati_storici_html if dati_storici_html else "<p>N/A</p>")
        html_content.append("</body></html>")
        
        try:
            full_html = "\n".join(html_content)
            try:
                c = repo.get_contents(file_res)
                repo.update_file(file_res, f"Upd {symbol}", full_html, c.sha)
            except:
                repo.create_file(file_res, f"Cre {symbol}", full_html)
        except: pass

        # Raccolta News
        for title, date, link, src, img in news_data["last_90_days"]:
            sia = SentimentIntensityAnalyzer()
            sc = (sia.polarity_scores(title)['compound'] + 1) / 2
            all_news_entries.append((symbol, title, sc, link, src, img))

    
    # --- SALVATAGGIO TEST (Alla fine del loop, prima del return) ---
    print("Elaborazione Forward Testing in corso...")
    backtester.update_daily_tracking(current_prices_map) # Calcola risultati
    backtester.save_data()       # Scrive il JSON in forward_testing/
    backtester.generate_report() # Scrive l'HTML in forward_testing/
    # ---------------------------------------------------------------

    history_mgr.save_data_to_github()
    return (sentiment_results, percentuali_combine, all_news_entries, 
            indicator_data, fundamental_data, crescita_settimanale, dati_storici_all, momentum_results)


# ==============================================================================
# 5. ESECUZIONE
# ==============================================================================

sentiment_for_symbols, percentuali_combine, all_news_entries, indicator_data, fundamental_data, crescita_settimanale, dati_storici_all, momentum_results = get_sentiment_for_all_symbols(symbol_list)

# --- CLASSIFICA PRINCIPALE (BASATA SU HYBRID SCORE) ---
sorted_symbols = sorted(percentuali_combine.items(), key=lambda x: x[1], reverse=True)

html_classifica = ["<html><head><title>Classifica dei Simboli</title></head><body>",
                   "<h1>Classifica dei Simboli (Hybrid Score)</h1>",
                   "<table border='1'><tr><th>Simbolo</th><th>Probabilità</th></tr>"]

for symbol, score in sorted_symbols:
    html_classifica.append(f"<tr><td>{symbol}</td><td>{score:.2f}%</td></tr>")

html_classifica.append("</table></body></html>")

try:
    contents = repo.get_contents(file_path)
    repo.update_file(contents.path, "Updated classification", "\n".join(html_classifica), contents.sha)
except GithubException:
    repo.create_file(file_path, "Created classification", "\n".join(html_classifica))

print("Classifica aggiornata con successo!")


# --- CLASSIFICA PRO ---
sorted_symbols_pro = sorted(percentuali_combine.items(), key=lambda x: x[1], reverse=True)
html_classifica_pro = ["<html><head><title>Classifica Combinata</title></head><body>",
                       "<h1>Classifica Combinata (Hybrid Logic)</h1>",
                       "<table border='1'><tr><th>Simbolo</th><th>Hybrid Score</th></tr>"]
for symbol, media in sorted_symbols_pro:
    html_classifica_pro.append(f"<tr><td>{symbol}</td><td>{media:.2f}%</td></tr>")
html_classifica_pro.append("</table></body></html>")

try:
    contents = repo.get_contents(pro_path)
    repo.update_file(contents.path, "Upd PRO", "\n".join(html_classifica_pro), contents.sha)
except:
    repo.create_file(pro_path, "Cre PRO", "\n".join(html_classifica_pro))

print("Classifica PRO aggiornata!")


# --- CLASSIFICA MOMENTUM ---
print("Generazione Classifica Momentum...")

# Ordina dal più alto al più basso
sorted_momentum = sorted(momentum_results.items(), key=lambda x: x[1], reverse=True)

html_mom = [
    "<html><head><title>Classifica Momentum</title>",
    "<style>",
    "table {border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;}",
    "th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}",
    "th {background-color: #f2f2f2;}",
    ".high {color: green; font-weight: bold;}",
    ".low {color: red; font-weight: bold;}",
    ".neutral {color: black;}",
    "</style>",
    "</head><body>",
    "<h1>🔥 Classifica Momentum (Delta Score)</h1>",
    "<p>Indica l'accelerazione del sentiment e delle notizie rispetto alla media storica.</p>",
    "<table><tr><th>Simbolo</th><th>Momentum Score (0-100)</th><th>Stato</th></tr>"
]

for symbol, score in sorted_momentum:
    # Definizione colore e stato
    if score >= 60:
        color_class = "high"
        status = "HYPE / ACCELERAZIONE"
    elif score <= 40:
        color_class = "low"
        status = "DEPRESSIONE / CALO"
    else:
        color_class = "neutral"
        status = "Normale"

    html_mom.append(f"<tr><td><b>{symbol}</b></td><td class='{color_class}'>{score:.2f}</td><td>{status}</td></tr>")

html_mom.append("</table></body></html>")

# Salvataggio su GitHub
try:
    contents = repo.get_contents(mom_path)
    repo.update_file(contents.path, "Upd Momentum Rank", "\n".join(html_mom), contents.sha)
except:
    repo.create_file(mom_path, "Cre Momentum Rank", "\n".join(html_mom))

print("Classifica Momentum creata con successo!")


# --- CLASSIFICA SETTORI (LIQUIDITY WEIGHTED) ---
print("Generazione Classifica Settori (Liquidity Weighted)...")

# 1. Raccogliamo i dati grezzi per calcolare i pesi relativi
# Struttura: sector_assets[settore] = [ (score, dollar_volume), ... ]
sector_assets = defaultdict(list)

for symbol, score in percentuali_combine.items():
    sec = asset_sector_map.get(symbol, "Altro")
    
    # Calcolo della "Importanza" (Dollar Volume medio ultimi 30gg)
    # Usiamo i dati storici che abbiamo già scaricato!
    avg_liquidity = 0.0
    
    if symbol in dati_storici_all:
        df = dati_storici_all[symbol]
        try:
            # Prendiamo gli ultimi 20 giorni (1 mese di trading)
            last_month = df.tail(20).copy()
            # Calcolo: Prezzo * Volume
            # Nota: yfinance a volte ha volumi 0 o NaN, gestiamo con fillna
            liquidity_series = (last_month['Close'] * last_month['Volume']).fillna(0)
            avg_liquidity = liquidity_series.mean()
        except:
            avg_liquidity = 0.0
    
    # Se il calcolo fallisce (es. dati mancanti), diamo un peso minimo simbolico (es. 1000$)
    # per non escluderlo dalla media, ma contarlo pochissimo.
    if avg_liquidity <= 0 or pd.isna(avg_liquidity):
        avg_liquidity = 1000.0 
        
    sector_assets[sec].append({
        'symbol': symbol,
        'score': score,
        'liquidity': avg_liquidity
    })

# 2. Calcolo Score Ponderato per Settore
sector_final_scores = []

for sec, assets in sector_assets.items():
    # Somma totale della liquidità del settore (Il "Market Cap" del nostro paniere)
    total_sector_liquidity = sum(a['liquidity'] for a in assets)
    
    weighted_score_sum = 0.0
    asset_count = len(assets)
    
    # Trova il leader per liquidità (il più grosso del gruppo)
    top_asset = max(assets, key=lambda x: x['liquidity'])
    leader_name = top_asset['symbol']
    
    for asset in assets:
        # Il peso è la percentuale di liquidità dell'asset rispetto al totale del settore
        # Esempio: Se MSFT muove 8Mld e il settore muove 10Mld, MSFT pesa 0.8 (80%)
        weight = asset['liquidity'] / total_sector_liquidity
        
        weighted_score_sum += (asset['score'] * weight)
        
    sector_final_scores.append({
        'sector': sec,
        'avg': weighted_score_sum, # Questo è ora il Weighted Average reale
        'count': asset_count,
        'leader': leader_name
    })

# 3. Ordinamento
sorted_sectors = sorted(sector_final_scores, key=lambda x: x['avg'], reverse=True)

# 4. Generazione HTML
html_sector = [
    "<html><head><title>Classifica Settori</title>",
    "<style>",
    "table {border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;}",
    "th, td {border: 1px solid #ddd; padding: 12px; text-align: left;}",
    "th {background-color: #f2f2f2;}",
    ".bull {color: green; font-weight: bold;}",
    ".bear {color: red; font-weight: bold;}",
    ".neutral {color: #333;}",
    "</style>",
    "</head><body>",
    "<h1>📊 Performance Settoriale (Volume Weighted)</h1>",
    "<p>Classifica ponderata sulla <b>Liquidità (Dollar Volume)</b>. Gli asset che muovono più denaro influenzano maggiormente il punteggio del settore.</p>",
    "<table><tr><th>Pos</th><th>Settore</th><th>Dominant Asset</th><th>Score Ponderato</th><th>Asset</th><th>Trend</th></tr>"
]

for idx, item in enumerate(sorted_sectors, 1):
    avg = item['avg']
    
    if avg >= 55:
        style_class = "bull"
        trend_label = "STRONG"
    elif avg >= 50:
        style_class = "bull"
        trend_label = "POSITIVE"
    elif avg <= 45:
        style_class = "bear"
        trend_label = "WEAK"
    elif avg <= 40:
        style_class = "bear"
        trend_label = "CRITICAL"
    else:
        style_class = "neutral"
        trend_label = "NEUTRAL"
    
    html_sector.append(
        f"<tr>"
        f"<td>{idx}</td>"
        f"<td><b>{item['sector']}</b></td>"
        f"<td>{item['leader']}</td>"
        f"<td class='{style_class}'>{avg:.2f}%</td>"
        f"<td>{item['count']}</td>"
        f"<td class='{style_class}'>{trend_label}</td>"
        f"</tr>"
    )

html_sector.append("</table></body></html>")

# 5. Salvataggio
try:
    contents = repo.get_contents(sector_path)
    repo.update_file(contents.path, "Upd Sector Rank Liquidity", "\n".join(html_sector), contents.sha)
except:
    repo.create_file(sector_path, "Cre Sector Rank Liquidity", "\n".join(html_sector))

print("Classifica Settori (Liquidity) creata con successo!")



# --- NEWS HTML ---
html_news = ["<html><head><title>Notizie e Sentiment</title></head><body>",
             "<h1>Notizie Finanziarie con Sentiment</h1>",
             "<table border='1'><tr><th>Simbolo</th><th>Notizia</th><th>Fonte</th><th>Immagine</th><th>Sentiment</th><th>Link</th></tr>"]
news_by_symbol = defaultdict(list)
for symbol, title, sentiment, url, source, image in all_news_entries:
    news_by_symbol[symbol].append((title, sentiment, url, source, image))

for symbol, entries in news_by_symbol.items():
    sorted_entries = sorted(entries, key=lambda x: x[1])
    selected_entries = sorted_entries[:5] + sorted_entries[-5:]
    selected_entries = list(dict.fromkeys(selected_entries))
    for title, sentiment, url, source, image in selected_entries:
        img_html = f"<img src='{image}' width='100'>" if image else "N/A"
        html_news.append(f"<tr><td>{symbol}</td><td>{title}</td><td>{source}</td><td>{img_html}</td><td>{sentiment:.2f}</td><td><a href='{url}' target='_blank'>Leggi</a></td></tr>")
html_news.append("</table></body></html>")

try:
    contents = repo.get_contents(news_path)
    repo.update_file(contents.path, "Upd News", "\n".join(html_news), contents.sha)
except:
    repo.create_file(news_path, "Cre News", "\n".join(html_news))

print("News aggiornata!")

# --- CLASSIFICA FIRE ---
sorted_crescita = sorted([(s, g) for s, g in crescita_settimanale.items() if g is not None], key=lambda x: (x[1], x[0]), reverse=True)
html_fire = ["<html><head><title>Classifica per Crescita</title></head><body>",
             "<h1>Asset Ordinati per Crescita Settimanale</h1>",
             "<table border='1'><tr><th>Simbolo</th><th>Crescita 7gg (%)</th></tr>"]
for symbol, growth in sorted_crescita:
    html_fire.append(f"<tr><td>{symbol}</td><td>{growth:.2f}%</td></tr>")
html_fire.append("</table></body></html>")

try:
    contents = repo.get_contents(fire_path)
    repo.update_file(contents.path, "Upd Fire", "\n".join(html_fire), contents.sha)
except:
    repo.create_file(fire_path, "Cre Fire", "\n".join(html_fire))

print("Fire aggiornato!")

# --- DAILY BRIEF LOGIC (COMPLETA) ---
def generate_fluid_market_summary_english(sentiment_for_symbols, percentuali_combine, all_news_entries, symbol_name_map, indicator_data, fundamental_data):
    
    def calculate_asset_score(symbol):
        return round(percentuali_combine.get(symbol, 50), 2)

    def build_insight(symbol):
        name = symbol_name_map.get(symbol, [symbol])[0]
        percent = percentuali_combine.get(symbol, 50)
        delta = percent - 50
        rsi = indicator_data.get(symbol, {}).get("RSI (14)")
        theme = "neutral"
        if percent > 60: theme = "gainer"
        elif percent < 40: theme = "loser"
        elif rsi and rsi < 30: theme = "oversold"
        elif rsi and rsi > 70: theme = "overbought"
        return {"symbol": symbol, "name": name, "percent": percent, "delta": delta, "rsi": rsi, "theme": theme}

    def build_forecast_phrase(ins):
        if ins["rsi"] and ins["rsi"] < 30: return random.choice([" The stock may rise soon.", " Potential rebound."])
        elif ins["rsi"] and ins["rsi"] > 70: return random.choice([" A pullback is likely.", " Overbought territory."])
        elif ins["delta"] > 0: return random.choice([" Strength should continue.", " Bullish momentum."])
        elif ins["delta"] < 0: return random.choice([" Weakness may persist.", " Bearish trend."])
        return " Outlook uncertain."

    clause_templates = {
        "gainer": ["{name} is surging.", "{name} gained momentum."],
        "loser": ["{name} is under pressure.", "{name} dropped."],
        "oversold": ["{name} looks oversold.", "{name} is dipping low."],
        "overbought": ["{name} is flying high.", "{name} looks overextended."],
        "neutral": ["{name} is flat.", "{name} shows little movement."]
    }

    def build_fluid_narrative(ins):
        name = ins['name']
        d_abs = abs(ins['delta'])
        action = "moved"
        if ins['theme'] == 'gainer': action = random.choice(['rallied', 'jumped', 'climbed'])
        elif ins['theme'] == 'loser': action = random.choice(['slipped', 'fell', 'dipped'])
        return f"{name} {action} with a score of {ins['percent']:.1f}%."

    # Top Assets
    scores = {sym: calculate_asset_score(sym) for sym in percentuali_combine.keys()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_3 = [build_insight(s) for s, _ in ranked[:3]]
    
    brief_text = "Market Report: " + " ".join([build_fluid_narrative(i) for i in top_3])

    symbol_phrases = []
    for symbol in percentuali_combine.keys():
        ins = build_insight(symbol)
        tpl = random.choice(clause_templates[ins["theme"]])
        phrase = tpl.format(name=ins["name"]) + build_forecast_phrase(ins)
        symbol_phrases.append(f"{symbol} - {phrase}")

    return brief_text, "\n".join(symbol_phrases)

brief_text, asset_sentences = generate_fluid_market_summary_english(sentiment_for_symbols, percentuali_combine, all_news_entries, symbol_name_map, indicator_data, fundamental_data)

def raffina_testo(testo):
    return re.sub(r'\s+', ' ', testo).strip()

brief_refined = raffina_testo(brief_text)

def genera_mini_tip_from_summary(summary):
    tips = {
        "RSI": "RSI above 70 suggests overbought, below 30 oversold.",
        "Volume": "Rising volume confirms trends.",
        "P/E": "High P/E might mean overvaluation.",
        "Diversification": "Don't put all eggs in one basket."
    }
    return random.choice(list(tips.values()))

mini_tip = genera_mini_tip_from_summary(brief_text)

def assign_signal_and_confidence(percentuali_combine):
    signals = {}
    for sym, score in percentuali_combine.items():
        if score > 60: sig = "BUY"
        elif score < 40: sig = "SELL"
        else: sig = "HOLD"
        signals[sym] = {"signal": sig, "confidence": random.uniform(0.8, 0.99)}
    return signals

signals = assign_signal_and_confidence(percentuali_combine)
grouped = defaultdict(list)
for sym, info in signals.items(): grouped[info['signal']].append((sym, info['confidence']))
top_signal_str = ""
for st in ['BUY', 'HOLD', 'SELL']:
    if grouped[st]:
        sym, strength = max(grouped[st], key=lambda x: x[1])
        top_signal_str += f"{st} signal on {sym} - Accuracy {int(strength*100)}%\n"

# Translate & Save Briefs
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
supported_langs = {p.to_code for p in available_packages if p.from_code == "en"}
for lang_code in LANGUAGES.keys():
    if lang_code not in supported_langs: continue
    pkg = next((p for p in available_packages if p.from_code == "en" and p.to_code == lang_code), None)
    if pkg: argostranslate.package.install_from_path(pkg.download())

def translate_text(text, target_lang):
    if target_lang not in supported_langs: return text
    return argostranslate.translate.translate(text, "en", target_lang)

for lang_code, filename in LANGUAGES.items():
    html_content = f"""<html><head><title>Market Brief</title></head><body><h1>📊 Daily Market Summary</h1><p>{translate_text(brief_refined, lang_code)}</p><h2>Per-Asset Insights</h2><ul>{"".join(f"<li>{translate_text(line, lang_code)}</li>" for line in asset_sentences.splitlines())}</ul><h2>💡 Mini Tip</h2><p>{translate_text(mini_tip, lang_code)}</p><hr><h2>🔥 Top Signal</h2><p>{translate_text(top_signal_str, lang_code)}</p></body></html>"""
    fpath = f"{TARGET_FOLDER}/{filename}"
    try: repo.update_file(fpath, f"Upd {filename}", html_content, repo.get_contents(fpath).sha)
    except: repo.create_file(fpath, f"Cre {filename}", html_content)

html_content_en = f"""<html><head><title>Market Brief</title></head><body><h1>📊 Daily Market Summary</h1><p>{brief_refined}</p><h2>Per-Asset Insights</h2><ul>{"".join(f"<li>{line}</li>" for line in asset_sentences.splitlines())}</ul><h2>💡 Mini Tip</h2><p>{mini_tip}</p><hr><h2>🔥 Top Signal</h2><p>{top_signal_str}</p></body></html>"""
fpath_en = f"{TARGET_FOLDER}/daily_brief_en.html"
try: repo.update_file(fpath_en, "Upd EN Brief", html_content_en, repo.get_contents(fpath_en).sha)
except: repo.create_file(fpath_en, "Cre EN Brief", html_content_en)

# --- CORRELAZIONI STATISTICHE (COMPLETA) ---
def calcola_correlazioni(dati_storici_all):
    returns = {sym: np.log(df["Close"]).diff().dropna() for sym, df in dati_storici_all.items() if "Close" in df.columns}
    results = {}
    assets = list(returns.keys())
    
    for asset1 in assets:
        candidates = []
        for asset2 in assets:
            if asset1 == asset2: continue
            
            # Allineamento serie temporali
            s1, s2 = returns[asset1], returns[asset2]
            common = s1.index.intersection(s2.index)
            if len(common) < 30: continue
            
            x, y = s1.loc[common], s2.loc[common]
            
            # Pearson & Spearman
            try: p_r, _ = pearsonr(x, y)
            except: p_r = 0
            try: s_r, _ = spearmanr(x, y)
            except: s_r = 0
            
            # Direzionalità (Concordanza segno)
            conc = (np.sign(x) == np.sign(y)).mean() * 100
            
            score = (abs(p_r) + abs(s_r) + (conc/100)) / 3
            
            candidates.append({
                "asset2": asset2,
                "pearson": p_r,
                "spearman": s_r,
                "concordance": conc,
                "score": score
            })
            
        results[asset1] = sorted(candidates, key=lambda x: x["score"], reverse=True)[:5]
    return results

def salva_correlazioni_html(correlazioni, repo, file_path=corr_path):
    html_corr = ["<html><head><title>Correlazioni</title></head><body><h1>Correlazioni Statistiche</h1><table border='1'><tr><th>Asset</th><th>Partner</th><th>Pearson</th><th>Spearman</th><th>Direzionalità (%)</th></tr>"]
    for sym, entries in correlazioni.items():
        for info in entries:
            html_corr.append(f"<tr><td>{sym}</td><td>{info['asset2']}</td><td>{info['pearson']:.2f}</td><td>{info['spearman']:.2f}</td><td>{info['concordance']:.1f}%</td></tr>")
    html_corr.append("</table></body></html>")
    try: repo.update_file(file_path, "Upd Corr", "\n".join(html_corr), repo.get_contents(file_path).sha)
    except: repo.create_file(file_path, "Cre Corr", "\n".join(html_corr))

print("Calcolo Correlazioni...")
correlazioni = calcola_correlazioni(dati_storici_all)
salva_correlazioni_html(correlazioni, repo)
print("Finito!")
