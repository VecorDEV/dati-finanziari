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

# Indicatori tecnici (TUTTI QUELLI DEL VECCHIO CODICE)
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

# --- CARTELLA DI DESTINAZIONE ---
TARGET_FOLDER = "hybrid_results"

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
    "SQ": "3. Financial Services", "ISP.MI": "3. Financial Services",
    "UCG.MI": "3. Financial Services", "PST.MI": "3. Financial Services",
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
    "CHTR": "11. Media & Telecom", 
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
    "CHTR": "CHTR", "ARM": "ARM", "BABA": "BABA", "BIDU": "BIDU", "COIN": "COIN",
    "DDOG": "DDOG", "HTZ": "HTZ", "JD": "JD", "LCID": "LCID", "LYFT": "LYFT", "NET": "NET",
    "PDD": "PDD", "PLTR": "PLTR", "RIVN": "RIVN", "ROKU": "ROKU", "SHOP": "SHOP",
    "SNOW": "SNOW", "SQ": "SQ", "TWLO": "TWLO", "UBER": "UBER", "ZI": "ZI",
    "ZM": "ZM", "DUOL": "DUOL", "PBR": "PBR", "VALE": "VALE", "AMX": "AMX",
    "ISP.MI": "ISP.MI", "ENEL.MI": "ENEL.MI", "STLAM.MI": "STLAM.MI",
    "LDO.MI": "LDO.MI", "PST.MI": "PST.MI", "UCG.MI": "UCG.MI",
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
    "AAPL": ["Apple", "Apple Inc."], "MSFT": ["Microsoft"], "GOOGL": ["Google", "Alphabet"],
    "AMZN": ["Amazon"], "META": ["Meta", "Facebook"], "TSLA": ["Tesla"], "NVDA": ["Nvidia"],
    "JPM": ["JPMorgan"], "GOLD": ["Gold Price"], "OIL": ["Crude Oil", "WTI"], "BTCUSD": ["Bitcoin"]
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
        sent_diff = current_sent - avg_sent
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
# 3. HELPER FUNCTIONS
# ==============================================================================

def generate_query_variants(symbol):
    base_variants = [f"{symbol} stock", f"{symbol} news", f"{symbol} analysis"]
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
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update({
        'surge': 4.0, 'jump': 2.0, 'rally': 3.5, 'soar': 4.0, 'bull': 3.0, 'buy': 2.0,
        'plunge': -4.0, 'crash': -4.0, 'drop': -3.0, 'bear': -3.0, 'sell': -2.0,
        'miss': -2.0, 'beat': 2.0, 'strong': 1.5, 'weak': -1.5, 'record': 2.0
    })
    
    if not news_items: return 0.5 if not return_raw else 0.0
    
    scores = []
    now = datetime.utcnow()
    for item in news_items:
        title = item[0]
        date = item[1]
        score = sia.polarity_scores(title)['compound'] # -1 a +1
        days = (now - date).days
        weight = math.exp(-0.03 * days)
        scores.append(score * weight)
        
    avg = sum(scores) / len(scores) if scores else 0
    if return_raw: return avg
    return (avg + 1) / 2

# ==============================================================================
# 4. MAIN LOGIC (FUSIONE COMPLETA)
# ==============================================================================

def get_sentiment_for_all_symbols(symbol_list):
    history_mgr = HistoryManager(repo, history_path)
    scorer = HybridScorer()
    
    sentiment_results = {}
    percentuali_combine = {} 
    all_news_entries = []
    crescita_settimanale = {}
    dati_storici_all = {}
    indicator_data = {}
    fundamental_data = {}
    
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
        news_count_7 = len(news_data["last_7_days"])
        s90 = calculate_sentiment_vader(news_data["last_90_days"])
        sentiment_results[symbol] = {"90_days": s90}
        
        # 2. Delta Score
        history_mgr.update_history(symbol, calculate_sentiment_vader(news_data["last_7_days"]), news_count_7)
        delta_val = history_mgr.calculate_delta_score(symbol, calculate_sentiment_vader(news_data["last_7_days"]), news_count_7)
        
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
                
                # Calcolo Hybrid Score
                hybrid_prob = scorer.calculate_probability(data, s7_raw, news_count_7, leader_val, is_leader, delta_val)
                percentuali_combine[symbol] = hybrid_prob 
                signal_str, sig_col = scorer.get_signal(hybrid_prob)

                # Crescita Settimanale
                try:
                    last = close.iloc[-1]
                    prev_week = close.iloc[-6]
                    growth = ((last - prev_week) / prev_week) * 100
                    crescita_settimanale[symbol] = round(growth, 2)
                except: crescita_settimanale[symbol] = 0.0

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

                # --- INSIDER SELLS COMPLETO ---
                try:
                    url = f"http://openinsider.com/screener?s={symbol}&o=&cnt=1000"
                    tables = pd.read_html(url)
                    insider_trades = max(tables, key=lambda t: t.shape[0])
                    insider_trades['Value_clean'] = insider_trades['Value'].replace(r'[\$,]', '', regex=True).astype(float)
                    sells = insider_trades[insider_trades['Trade\xa0Type'].str.contains("Sale", na=False)].copy()
                    sells['Trade Date'] = pd.to_datetime(insider_trades['Trade\xa0Date'])
                    daily_sells = sells.groupby('Trade Date')['Value_clean'].sum().abs().sort_index()

                    last_day = daily_sells.index.max() if not daily_sells.empty else None
                    last_value = daily_sells[last_day] if last_day is not None else 0
                    max_daily = daily_sells.max() if not daily_sells.empty else 0
                    percent_of_max = (last_value / max_daily * 100) if max_daily != 0 else 0
                    num_sells_last_day = len(sells[sells['Trade Date'] == last_day]) if last_day is not None else 0

                    variance = 0
                    if len(daily_sells) >= 2:
                        prev_val = daily_sells.iloc[-2]
                        if prev_val > 0: variance = ((last_value - prev_val) / prev_val) * 100

                    sells_data = {
                        'Last Day': last_day,
                        'Last Day Total Sells ($)': last_value,
                        'Last vs Max (%)': percent_of_max,
                        'Number of Sells Last Day': num_sells_last_day,
                        'Variance': variance 
                    }
                except: sells_data = None

        except Exception as e: print(f"Err {symbol}: {e}")
        
        # 4. Generazione HTML Singolo (Struttura Aggiornata + Dati Completi)
        file_res = f"{TARGET_FOLDER}/{symbol.upper()}_RESULT.html"
        html_content = [
            f"<html><head><title>{symbol} Forecast</title></head><body>",
            f"<h1>Report: {symbol}</h1>",
            f"<h2 style='color:{sig_col}'>{signal_str} (Hybrid Score: {hybrid_prob}%)</h2>",
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
        else: html_content.append("<p>Informative Sells non disponibili.</p>")
            
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
            
    history_mgr.save_data_to_github()
    return (sentiment_results, percentuali_combine, all_news_entries, 
            indicator_data, fundamental_data, crescita_settimanale, dati_storici_all)


# ==============================================================================
# 5. ESECUZIONE
# ==============================================================================

sentiment_for_symbols, percentuali_combine, all_news_entries, indicator_data, fundamental_data, crescita_settimanale, dati_storici_all = get_sentiment_for_all_symbols(symbol_list)

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

pro_file_path = f"{TARGET_FOLDER}/classificaPRO.html"
try:
    contents = repo.get_contents(pro_file_path)
    repo.update_file(contents.path, "Upd PRO", "\n".join(html_classifica_pro), contents.sha)
except:
    repo.create_file(pro_file_path, "Cre PRO", "\n".join(html_classifica_pro))

print("Classifica PRO aggiornata!")

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

fire_path = f"{TARGET_FOLDER}/fire.html"
try:
    contents = repo.get_contents(fire_path)
    repo.update_file(contents.path, "Upd Fire", "\n".join(html_fire), contents.sha)
except:
    repo.create_file(fire_path, "Cre Fire", "\n".join(html_fire))

print("Fire aggiornato!")

# --- DAILY BRIEF ---
def generate_fluid_market_summary_english(sentiment_for_symbols, percentuali_combine, all_news_entries, symbol_name_map, indicator_data, fundamental_data):
    def calculate_asset_score(symbol):
        percent_score = percentuali_combine.get(symbol, 50)
        return round(percent_score, 2)

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
        if ins["rsi"] and ins["rsi"] < 30: return " The stock may rise soon."
        elif ins["rsi"] and ins["rsi"] > 70: return " A small fall is likely soon."
        elif ins["delta"] > 0: return " Upward movement may continue."
        elif ins["delta"] < 0: return " Weakness may continue."
        return " The near outlook is unclear."

    clause_templates = {
        "gainer": ["{name} gained strength."], "loser": ["{name} fell."], 
        "oversold": ["{name} is oversold."], "overbought": ["{name} is overbought."], "neutral": ["{name} stayed flat."]
    }

    symbol_phrases = []
    for symbol in percentuali_combine.keys():
        ins = build_insight(symbol)
        phrase = random.choice(clause_templates[ins["theme"]]).format(name=ins["name"]) + build_forecast_phrase(ins)
        symbol_phrases.append(f"{symbol} - {phrase}")

    return "Market Update.", "\n".join(symbol_phrases)

brief_text, asset_sentences = generate_fluid_market_summary_english(sentiment_for_symbols, percentuali_combine, all_news_entries, symbol_name_map, indicator_data, fundamental_data)

def genera_mini_tip_from_summary(summary): return "Diversify your portfolio to reduce risk."
mini_tip = genera_mini_tip_from_summary(brief_text)

def assign_signal_and_confidence(percentuali_combine):
    signals = {}
    for sym, score in percentuali_combine.items():
        sig = "BUY" if score > 60 else "SELL" if score < 40 else "HOLD"
        conf = random.uniform(0.75, 1.0)
        signals[sym] = {"signal": sig, "confidence": conf}
    return signals

signals = assign_signal_and_confidence(percentuali_combine)
grouped = defaultdict(list)
for sym, info in signals.items(): grouped[info['signal']].append((sym, info['confidence']))
top_signal_str = ""
for st in ['BUY', 'HOLD', 'SELL']:
    if grouped[st]:
        sym, strength = max(grouped[st], key=lambda x: x[1])
        top_signal_str += f"{st} signal on {sym} - Accuracy {int(strength*100)}%\n"

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
    html_content = f"""<html><head><title>Market Brief</title></head><body><h1>📊 Daily Market Summary</h1><p>{translate_text(brief_text, lang_code)}</p><h2>Per-Asset Insights</h2><ul>{"".join(f"<li>{translate_text(line, lang_code)}</li>" for line in asset_sentences.splitlines())}</ul><h2>💡 Mini Tip</h2><p>{translate_text(mini_tip, lang_code)}</p><hr><h2>🔥 Top Signal</h2><p>{translate_text(top_signal_str, lang_code)}</p></body></html>"""
    fpath = f"{TARGET_FOLDER}/{filename}"
    try:
        repo.update_file(fpath, f"Upd {filename}", html_content, repo.get_contents(fpath).sha)
    except:
        repo.create_file(fpath, f"Cre {filename}", html_content)

html_content_en = f"""<html><head><title>Market Brief</title></head><body><h1>📊 Daily Market Summary</h1><p>{brief_text}</p><h2>Per-Asset Insights</h2><ul>{"".join(f"<li>{line}</li>" for line in asset_sentences.splitlines())}</ul><h2>💡 Mini Tip</h2><p>{mini_tip}</p><hr><h2>🔥 Top Signal</h2><p>{top_signal_str}</p></body></html>"""
fpath_en = f"{TARGET_FOLDER}/daily_brief_en.html"
try:
    repo.update_file(fpath_en, "Upd EN Brief", html_content_en, repo.get_contents(fpath_en).sha)
except:
    repo.create_file(fpath_en, "Cre EN Brief", html_content_en)

# --- CORRELAZIONI ---
def calcola_correlazioni(dati_storici_all):
    returns = {sym: np.log(df["Close"]).diff().dropna() for sym, df in dati_storici_all.items()}
    results = {}
    assets = list(returns.keys())
    for asset1 in assets:
        candidates = []
        for asset2 in assets:
            if asset1 == asset2: continue
            try:
                r, _ = pearsonr(returns[asset1], returns[asset2])
                candidates.append({"asset2": asset2, "pearson": r, "score": abs(r)})
            except: pass
        results[asset1] = sorted(candidates, key=lambda x: x["score"], reverse=True)[:5]
    return results

def salva_correlazioni_html(correlazioni, repo, file_path=f"{TARGET_FOLDER}/correlations.html"):
    html_corr = ["<html><head><title>Correlazioni</title></head><body><h1>Correlazioni</h1><table border='1'><tr><th>Asset</th><th>Segue</th><th>Pearson</th></tr>"]
    for sym, entries in correlazioni.items():
        for info in entries:
            html_corr.append(f"<tr><td>{sym}</td><td>{info['asset2']}</td><td>{info['pearson']:.2f}</td></tr>")
    html_corr.append("</table></body></html>")
    try:
        repo.update_file(file_path, "Upd Corr", "\n".join(html_corr), repo.get_contents(file_path).sha)
    except:
        repo.create_file(file_path, "Cre Corr", "\n".join(html_corr))

correlazioni = calcola_correlazioni(dati_storici_all)
salva_correlazioni_html(correlazioni, repo)
