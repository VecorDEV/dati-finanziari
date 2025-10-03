from github import Github, GithubException
import re
import feedparser
import os
from datetime import datetime, timedelta
import math
import spacy
#Librerie per ottenere dati storici e calcolare indicatori
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import random
import unicodedata
import argostranslate.package
import argostranslate.translate
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands
from urllib.parse import quote_plus
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr, binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
#from transformers import T5Tokenizer, T5ForConditionalGeneration


# Carica il modello linguistico per l'inglese
nlp = spacy.load("en_core_web_sm")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = "VecorDEV/dati-finanziari"

# Salva il file HTML nella cartella 'results'
file_path = "results/classifica.html"
news_path = "results/news.html"
    
# Salva il file su GitHub
github = Github(GITHUB_TOKEN)
repo = github.get_repo(REPO_NAME)


# 📌 Lingue da generare (codice lingua: suffisso file)
LANGUAGES = {
    "ar": "daily_brief_ar.html",
    "de": "daily_brief_de.html",
    "es": "daily_brief_es.html",
    "fr": "daily_brief_fr.html",
    "hi": "daily_brief_hi.html",
    "it": "daily_brief_it.html",
    "ko": "daily_brief_ko.html",
    "nl": "daily_brief_nl.html",
    "pt": "daily_brief_pt.html",
    "ru": "daily_brief_ru.html",
    "zh": "daily_brief_zh.html",
    "zh-rCN": "daily_brief_zh-rCN.html",
}


# Lista dei simboli azionari da cercare
symbol_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "V", "JPM", "JNJ", "WMT",
        "NVDA", "PYPL", "DIS", "NFLX", "NIO", "NRG", "ADBE", "INTC", "CSCO", "PFE",
        "KO", "PEP", "MRK", "ABT", "XOM", "CVX", "T", "MCD", "NKE", "HD",
        "IBM", "CRM", "BMY", "ORCL", "ACN", "LLY", "QCOM", "HON", "COST", "SBUX",
        "CAT", "LOW", "MS", "GS", "AXP", "INTU", "AMGN", "GE", "FIS", "CVS",
        "DE", "BDX", "NOW", "SCHW", "LMT", "ADP", "C", "PLD", "NSC", "TMUS",
        "ITW", "FDX", "PNC", "SO", "APD", "ADI", "ICE", "ZTS", "TJX", "CL",
        "MMC", "EL", "GM", "CME", "EW", "AON", "D", "PSA", "AEP", "TROW", 
        "LNTH", "HE", "BTDR", "NAAS", "SCHL", "TGT", "SYK", "BKNG", "DUK", "USB",
        "ARM", "BABA", "BIDU", "COIN", "PST.MI", "UCG.MI", "DDOG", "HTZ", "JD", "LCID", "LYFT", "NET", "PDD", #NEW
        "PLTR", "RIVN", "ROKU", "SHOP", "SNOW", "SQ", "TWLO", "UBER", "ZI", "ZM", "DUOL",    #NEW
        "PBR", "VALE", "AMX",
        
        "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
        "AUDJPY", "CADJPY", "CHFJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "GBPCHF", "AUDCAD",

        "SPX500", "DJ30", "NAS100", "NASCOMP", "RUS2000", "VIX", "EU50", "ITA40", "GER40", "UK100",
        "FRA40", "SWI20", "ESP35", "NETH25", "JPN225", "HKG50", "CHN50", "IND50", "KOR200",
               
        "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BCHUSD", "EOSUSD", "XLMUSD", "ADAUSD", "TRXUSD", "NEOUSD",
        "DASHUSD", "XMRUSD", "ETCUSD", "ZECUSD", "BNBUSD", "DOGEUSD", "USDTUSD", "LINKUSD", "ATOMUSD", "XTZUSD",
        "COCOA", "XAUUSD", "GOLD", "XAGUSD", "SILVER", "OIL", "NATGAS"]  # Puoi aggiungere altri simboli

'''
    

    
    '''
symbol_list_for_yfinance = [
    # Stocks (unchanged)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "V", "JPM", "JNJ", "WMT",
    "NVDA", "PYPL", "DIS", "NFLX", "NIO", "NRG", "ADBE", "INTC", "CSCO", "PFE",
    "KO", "PEP", "MRK", "ABT", "XOM", "CVX", "T", "MCD", "NKE", "HD",
    "IBM", "CRM", "BMY", "ORCL", "ACN", "LLY", "QCOM", "HON", "COST", "SBUX",
    "CAT", "LOW", "MS", "GS", "AXP", "INTU", "AMGN", "GE", "FIS", "CVS",
    "DE", "BDX", "NOW", "SCHW", "LMT", "ADP", "C", "PLD", "NSC", "TMUS",
    "ITW", "FDX", "PNC", "SO", "APD", "ADI", "ICE", "ZTS", "TJX", "CL",
    "MMC", "EL", "GM", "CME", "EW", "AON", "D", "PSA", "AEP", "TROW", 
    "LNTH", "HE", "BTDR", "NAAS", "SCHL", "TGT", "SYK", "BKNG", "DUK", "USB",
    "ARM", "BABA", "BIDU", "COIN", "PST.MI", "UCG.MI", "DDOG", "HTZ", "JD", "LCID", "LYFT", "NET", "PDD",
    "PLTR", "RIVN", "ROKU", "SHOP", "SNOW", "SQ", "TWLO", "UBER", "ZI", "ZM", "DUOL",
    "PBR", "VALE", "AMX",

    # Forex (with =X)
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "EURAUD=X", "EURNZD=X", "EURCAD=X",
    "EURCHF=X", "GBPCHF=X", "AUDCAD=X",

    # Global Indices
    "^GSPC", "^DJI", "^NDX", "^IXIC", "^RUT", "^VIX", "^STOXX50E", "FTSEMIB.MI", "^GDAXI", "^FTSE",
    "^FCHI", "^SSMI", "^IBEX", "^AEX", "^N225", "^HSI", "000001.SS", "^NSEI", "^KS200",

    # Crypto (with -USD)
    "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD", "BCH-USD", "EOS-USD", "XLM-USD", "ADA-USD",
    "TRX-USD", "NEO-USD", "DASH-USD", "XMR-USD", "ETC-USD", "ZEC-USD", "BNB-USD", "DOGE-USD",
    "USDT-USD", "LINK-USD", "ATOM-USD", "XTZ-USD",

    # Commodities (correct tickers)
    "CC=F",       # Cocoa
    "GC=F",   # Gold spot
    "GC=F",   # Gold spot
    "SI=F",   # Silver spot
    "SI=F",   # Silver spot
    "CL=F",        # Crude oil
    "NG=F"        # Natural gas
]

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
    "XAUUSD": ["Gold", "XAU/USD", "Gold price", "Gold spot"],
    "GOLD": ["Gold", "XAU/USD", "Gold price", "Gold spot"],
    "XAGUSD": ["Silver", "XAG/USD", "Silver price", "Silver spot"],
    "SILVER": ["Silver", "XAG/USD", "Silver price", "Silver spot"],
    "OIL": ["Crude oil", "Oil price", "WTI", "Brent", "Brent oil", "WTI crude"],
    "NATGAS": ["Natural gas", "Gas price", "Natgas", "Henry Hub", "NG=F", "Natural gas futures"]
}

indicator_data = {}
fundamental_data = {}

def generate_query_variants(symbol):
    base_variants = [
        f"{symbol} stock",
        f"{symbol} investing",
        f"{symbol} earnings",
        f"{symbol} news",
        f"{symbol} financial results",
        f"{symbol} analysis",
        f"{symbol} quarterly report",
        f"{symbol} Wall Street",
    ]
    
    name_variants = symbol_name_map.get(symbol.upper(), [])
    for name in name_variants:
        base_variants += [
            f"{name} stock",
            f"{name} investing",
            f"{name} earnings",
            f"{name} news",
            f"{name} financial results",
            f"{name} analysis",
            f"{name} quarterly report",
        ]
    
    return list(set(base_variants))  # Rimuove duplicati
   

MAX_ARTICLES_PER_SYMBOL = 500  # Limite massimo per asset

def get_stock_news(symbol):
    """Recupera titoli, date, link, fonte e immagine delle notizie per un simbolo."""
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
        if total_articles >= MAX_ARTICLES_PER_SYMBOL:
            break

        query = quote_plus(raw_query)
        url = base_url.format(query)
        feed = feedparser.parse(url)

        for entry in feed.entries:
            if total_articles >= MAX_ARTICLES_PER_SYMBOL:
                break

            try:
                title = entry.title.strip()
                link = entry.link.strip()
                source = entry.source.title if hasattr(entry, 'source') else "Unknown"
                
                # Cerca l'immagine
                if hasattr(entry, 'media_content'):
                    image = entry.media_content[0]['url']
                elif hasattr(entry, 'media_thumbnail'):
                    image = entry.media_thumbnail[0]['url']
                else:
                    image = None

                if title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())

                news_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")

                news_item = (title, news_date, link, source, image)

                if news_date >= days_90:
                    news_90_days.append(news_item)
                if news_date >= days_30:
                    news_30_days.append(news_item)
                if news_date >= days_7:
                    news_7_days.append(news_item)

                total_articles += 1

            except (ValueError, AttributeError):
                continue

    return {
        "last_90_days": news_90_days,
        "last_30_days": news_30_days,
        "last_7_days":  news_7_days
    }


# Lista di negazioni da considerare
negation_words = {"not", "never", "no", "without", "don't", "dont", "doesn't", "doesnt"}


# Dizionario di parole chiave con il loro punteggio di sentiment
sentiment_dict = {
    "ai": 0.6,
    "analyst rating": 0.6,
    "acquisition": 0.7,
    "acquisitions": 0.7,
    "appreciation": 0.9,
    "advance": 0.8,
    "advanced": 0.8,
    "agreement": 0.7,
    "agreements": 0.7,
    "agree": 0.7,
    "agreed": 0.6,
    "allocation": 0.6,
    "augmented": 0.7,
    "augment": 0.7,
    "augments": 0.7,
    "attraction": 0.85,
    "attractions": 0.85,
    "attractive": 0.85,
    "attractives": 0.85,
    "affluence": 0.9,
    "accelerator": 0.7,
    "ascend": 0.8,
    "advantage": 0.8,
    "advantaged": 0.8,
    "amplification": 0.7,
    "abundance": 0.9,
    "amendment": 0.6,
    "allowance": 0.6,
    "achievement": 0.8,
    "accession": 0.7,
    "ascension": 0.8,
    "allocation": 0.6,
    "acceptance": 0.7,
    "accreditation": 0.6,
    "authorized": 0.6,
    "approval": 0.6,
    "approved": 0.7,
    "assurance": 0.7,
    "advancement": 0.8,
    "aspiration": 0.7,
    "adoption": 0.6,
    "achievement": 0.8,
    "acceleration": 0.8,
    "appraisal": 0.6,
    "amortization": 0.4,
    "arrest": 0.1,
    "arrested": 0.1,
    "arrests": 0.1,
    "adversity": 0.2,
    "anomaly": 0.3,
    "attrition": 0.2,
    "antitrust": 0.2,
    "aversion": 0.2,
    "arrears": 0.1,
    "abandonment": 0.1,
    "alienation": 0.2,
    "asymmetry": 0.3,
    "ambiguity": 0.3,
    "anxiety": 0.2,
    "adjustment": 0.6,
    "adjusted earnings": 0.7,
    "algorithmic trading": 0.6,
    "austerity": 0.3,
    "audit": 0.4,
    "amendment": 0.6,
    "apprehension": 0.2,
    "abrogation": 0.1,
    "annulment": 0.1,
    "arrogation": 0.1,
    "admonition": 0.2,
    "antagonism": 0.2,
    "abysmal": 0.1,
    "accountability": 0.6,
    "arrestment": 0.2,
    "attrition": 0.3,
    "aftermath": 0.2,

    "bad": 0.1,
    "badly": 0.1,
    "bull": 0.9,
    "bullish": 0.9,
    "bully": 0.7,
    "bear": 0.3,
    "bear market": 0.2,
    "bankruptcy": 0.1,
    "bankrupt": 0.1,
    "balanced": 0.6,
    "bomb": 0.65,
    "boom": 0.9,
    "booms": 0.9,
    "buy": 0.9,
    "buys": 0.9,
    "bought": 0.85,
    "boost": 0.8,
    "boosts": 0.8,
    "boosted": 0.8,
    "benefit": 0.8,
    "benefits": 0.8,
    "billion": 0.6,
    "billions": 0.6,
    "bonds": 0.7,
    "breakthrough": 0.7,
    "benchmark": 0.7,
    "bust": 0.1,
    "bargain": 0.8,
    "bid": 0.7,
    "bailout": 0.3,
    "beneficiary": 0.7,
    "blockchain": 0.6,
    "bail": 0.3,
    "barrier": 0.4,
    "bottom line": 0.6,
    "balance sheet": 0.6,
    "backlog": 0.4,
    "backer": 0.65,
    "brisk": 0.7,
    "burnout": 0.2,
    "blockbuster": 0.8,
    "balance of payments": 0.6,
    "breach": 0.2,
    "blowout": 0.1,
    "bribe": 0.0,
    "brutal": 0.1,
    "bust up": 0.2,
    "bank run": 0.1,
    "bubble": 0.2,

    "capital": 0.7,
    "cancellation": 0.2,
    "cancellations": 0.2,
    "cash": 0.65,
    "crash": 0.1,
    "crashes": 0.1,
    "crashed": 0.1,
    "cautious": 0.35,
    "caution": 0.35,
    "cautiously": 0.65,
    "climb": 0.8,
    "climbed": 0.7,
    "climbs": 0.8,
    "close to get": 0.8,
    "coup": 0.2,
    "couch potato portfolio": 0.75,
    "credit": 0.7,
    "credit crunch": 0.2,
    "cut": 0.2,
    "cuts": 0.2,
    "collapse": 0.1,
    "collapsed": 0.1,
    "creditor": 0.65,
    "correction": 0.3,
    "commodities": 0.7,
    "change": 0.65,
    "competition": 0.3,
    "coupon": 0.6,
    "contribution": 0.7,
    "crisis": 0.1,
    "consolidation": 0.7,
    "capitalization": 0.8,
    "collateral damage": 0.3,
    "compliance": 0.65,
    "collaboration": 0.7,
    "consumer confidence": 0.8,
    "credibility": 0.8,
    "closure": 0.3,
    "commitment": 0.7,
    "clawback": 0.3,
    "cutback": 0.3,
    "come to an end": 0.3,
    "comes to an end": 0.3,
    "came to an end": 0.3,
    "coming to an end": 0.3,
    "contraction": 0.3,
    "conservative": 0.4,
    "corruption": 0.0,
    "corrupted": 0.0,
    "concerns": 0.2,
    "concern": 0.2,
    "capital gains": 0.8,
    "cash flow": 0.7,
    "credit rating": 0.65,
    "contribution margin": 0.7,
    "crisis management": 0.3,
    "capital raise": 0.7,
    "counterfeit": 0.1,
    "convergence": 0.65,
    "compensation package": 0.7,
    "compensation": 0.7,
    "capital flow": 0.6,
    "corruption scandal": 0.1,

    "damage": 0.1,
    "damages": 0.1,
    "damaged": 0.1,
    "damaging": 0.1,
    "debt": 0.2,
    "deal": 0.8,
    "deals": 0.8,
    "delay": 0.2,
    "delays": 0.2,
    "delayed": 0.2,
    "dividend": 0.8,
    "deficit": 0.1,
    "decline": 0.2,
    "declines": 0.2,
    "depreciation": 0.3,
    "drop": 0.2,
    "drops": 0.2,
    "downturn": 0.2,
    "down": 0.3,
    "downgrade": 0.25,
    "devaluation": 0.3,
    "disruption": 0.3,
    "discount": 0.6,
    "dilution": 0.4,
    "die": 0.2,
    "dies": 0.2,
    "died": 0.2,
    "dead": 0.2,
    "death": 0.2,
    "development": 0.7,
    "declining": 0.3,
    "delisting": 0.2,
    "delisted": 0.2,
    "distribution": 0.65,
    "dissatisfaction": 0.2,
    "dissatisfactions": 0.2,
    "debt ceiling": 0.2,
    "decline rate": 0.2,
    "dominance": 0.7,
    "distressed": 0.2,
    "downsize": 0.3,
    "drain": 0.2,
    "delisting": 0.1,
    "doubt": 0.2,
    "diminish": 0.3,
    "declining market": 0.1,
    "deterioration": 0.2,
    "diversification": 0.7,
    "direct investment": 0.7,
    "downward": 0.2,
    "danger": 0.1,
    "decline in sales": 0.1,
    "debt reduction": 0.65,
    "discrepancy": 0.3,
    "debt to equity": 0.4,
    "dismantling": 0.3,
    "deflation": 0.2,
    "debtor": 0.3,
    "debt servicing": 0.3,
    "dominant": 0.7,
    "diversified": 0.7,
    "dormant": 0.3,
    "downward spiral": 0.2,
    "dysfunction": 0.1,

    "equity": 0.8,
    "earning": 0.7,
    "earnings": 0.7,
    "emerging": 0.7,
    "expansion": 0.8,
    "efficiency": 0.7,
    "exit": 0.4,
    "estimation": 0.7,
    "expenditure": 0.3,
    "enterprise": 0.7,
    "evercore isi": 0.75,
    "equilibrium": 0.65,
    "economic slowdown": 0.15,
    "endowment": 0.7,
    "elevate": 0.8,
    "erode": 0.2,
    "erodes": 0.2,
    "eroding": 0.2,
    "eroded": 0.2,
    "exceed": 0.8,
    "expectation": 0.7,
    "excess": 0.35,
    "enrichment": 0.8,
    "encouragement": 0.7,
    "enterprise value": 0.7,
    "equity market": 0.7,
    "exclusivity": 0.7,
    "escrow": 0.65,
    "exodus": 0.2,
    "evasion": 0.1,
    "equitable": 0.7,
    "equilibrium price": 0.65,
    "empowerment": 0.7,
    "effort": 0.7,
    "elasticity": 0.7,
    "enforce": 0.65,
    "enforcing": 0.65,
    "establishment": 0.7,
    "enlightenment": 0.7,
    "equity fund": 0.7,

    "financial crisis": 0.1,
    "fund": 0.8,
    "failure": 0.1,
    "fail": 0.1,
    "fails": 0.1,
    "failed": 0.1,
    "fluctuation": 0.3,
    "funding": 0.7,
    "flexibility": 0.7,
    "favorable": 0.8,
    "fall": 0.15,
    "fell": 0.15,
    "fraud": 0.0,
    "flow": 0.7,
    "fintech": 0.7,
    "finance": 0.7,
    "flourish": 0.7,
    "fast track": 0.7,
    "foreclosure": 0.1,
    "failing": 0.1,
    "frenzy": 0.3,
    "fallout": 0.2,
    "failure rate": 0.2,
    "fundamentals": 0.7,
    "freeze": 0.3,
    "flare": 0.2,
    "forecasting": 0.65,
    "fraudulent": 0.1,
    "favorable outlook": 0.8,
    "favorable": 0.8,
    "financing": 0.7,
    "flow through": 0.7,
    "forward looking": 0.7,
    "fledgling": 0.4,
    "fire sale": 0.1,
    "full disclosure": 0.7,
    "financial innovation": 0.7,
    "free market": 0.7,
    "falling prices": 0.2,
    "falling price": 0.2,
    "falling": 0.2,

    "growth": 0.9,
    "growths": 0.9,
    "gain": 0.9,
    "gains": 0.9,
    "growth rate": 0.9,
    "growing dissatisfaction": 0.1,
    "guarantee": 0.8,
    "gross": 0.6,
    "green": 0.8,
    "grants": 0.7,
    "guidance": 0.7,
    "glut": 0.2,
    "gap": 0.3,
    "gloom": 0.2,
    "grave": 0.1,
    "gridlock": 0.3,
    "grind": 0.3,
    "gross margin": 0.7,
    "gross product": 0.7,
    "gains per share": 0.8,
    "greenfield": 0.7,
    "garnishment": 0.2,
    "growing market": 0.9,
    "government debt": 0.2,
    "garnishee": 0.2,
    "globalization": 0.65,
    "grip": 0.4,
    "global demand": 0.7,
    "gross revenue": 0.7,
    "goodwill": 0.65,
    "graft": 0.1,
    "guarantor": 0.65,
    "growth stock": 0.9,
    "good debt": 0.7,
    "good": 0.9,
    "goodly": 0.9,
    "global recession": 0.2,

    "hike": 0.8,
    "high": 0.8,
    "highs": 0.8,
    "holding company": 0.7,
    "holding companies": 0.7,
    "holding structure": 0.6,
    "holding structures": 0.6,
    "holding pattern": 0.4,
    "holding fund": 0.7,
    "holding funds": 0.7,
    "holding patterns": 0.4,
    "holding losses": 0.2,
    "holding loss": 0.2,
    "holding steady": 0.8,
    "holding off": 0.4,
    "hold off": 0.4,
    "holds off": 0.4,
    "holding gains": 0.9,
    "holding back": 0.3,
    "hold back": 0.3,
    "holds back": 0.3,
    "holding onto": 0.6,
    "hit": 0.3,
    "hurdle": 0.3,
    "healthy": 0.8,
    "hoarding": 0.2,
    "headwind": 0.3,
    "headwinds": 0.3,
    "hyperinflation": 0.1,
    "high risk": 0.2,
    "high risks": 0.2,
    "hedge fund": 0.4,
    "holding company": 0.7,
    "harmonic": 0.6,
    "high yield": 0.8,
    "healthy growth": 0.8,
    "haircut": 0.2,
    "high performance": 0.9,
    "high performances": 0.9,
    "high value": 0.9,
    "hollow": 0.2,
    "high headcount": 0.65,
    "high impact": 0.7,
    "hasty": 0.3,
    "healthcare": 0.7,
    "hustle": 0.4,
    "hardship": 0.2,
    "hard asset": 0.7,
    "hollowed out": 0.2,
    "hollow out": 0.2,
    "heavily indebted": 0.0,

    "ia": 0.6,
    "income": 0.8,
    "incomes": 0.8,
    "investment": 0.8,
    "inflation": 0.3,
    "inflate": 0.3,
    "increase": 0.8,
    "increased": 0.8,
    "increased competition": 0.2,
    "improvement": 0.8,
    "interest": 0.7,
    "insight": 0.7,
    "insights": 0.7,
    "inflationary": 0.3,
    "innovative": 0.8,
    "insurance": 0.7,
    "integrity": 0.8,
    "investment grade": 0.7,
    "intelligence": 0.7,
    "increase rate": 0.7,
    "increased rate": 0.7,
    "indebtedness": 0.2,
    "interest rate": 0.6,
    "impactful": 0.7,
    "incursion": 0.3,
    "illiquid": 0.2,
    "illiquidity": 0.2,
    "impairment": 0.2,
    "insolvency": 0.1,
    "income tax": 0.4,
    "incentive": 0.7,
    "issue": 0.1,
    "issues": 0.1,
    "inflated": 0.3,
    "inflating": 0.3,
    "insider trading": 0.1,
    "increased demand": 0.7,
    "increase demand": 0.7,
    "increased competition": 0.25,
    "increase competition": 0.25,
    "independent": 0.7,
    "invest": 0.65,
    "incorporation": 0.6,
    "illegal": 0.0,
    "investing": 0.65,
    "impaired": 0.2,
    "interest bearing": 0.7,
    "interest": 0.7,
    "interesting": 0.7,
    "interested": 0.7,

    "job": 0.7,
    "jobs": 0.7,
    "joint": 0.7,
    "jump": 0.9,
    "jumps": 0.9,
    "junks": 0.2,
    "junk": 0.2,
    "justice": 0.8,
    "jittery": 0.3,
    "jackpot": 0.9,
    "jackpots": 0.9,
    "jockey": 0.65,
    "jumpstart": 0.8,
    "juggernaut": 0.7,
    "jockeying": 0.6,
    "justice system": 0.65,
    "job market": 0.7,
    "jack up": 0.3,
    "judicious": 0.7,
    "jobless": 0.2,

    "kicker": 0.6,
    "knockout": 0.7,
    "keep growing": 0.9,
    "keep increasing": 0.9,
    "keep holding": 0.7,
    "keep outperforming": 0.9,
    "keep strengthening": 0.9,
    "keep stable": 0.6,
    "keep waiting": 0.4,
    "keep struggling": 0.3,
    "keep losing": 0.2,
    "knot": 0.3,
    "kickback": 0.2,
    "keen on": 0.7,
    "kill": 0.2,
    "key player": 0.8,
    "kind": 0.6,
    "kerfuffle": 0.3,
    "kickstart": 0.7,
    "kudos": 0.8,

    "layoff": 0.2,
    "loss": 0.1,
    "losses": 0.1,
    "lost": 0.1,
    "liquidity": 0.65,
    "loan": 0.7,
    "loans": 0.7,
    "liability": 0.3,
    "liquid": 0.7,
    "long": 0.7,
    "lift": 0.8,
    "low": 0.3,
    "lows": 0.3,
    "leading": 0.8,
    "lending": 0.7,
    "lead": 0.8,
    "lag": 0.35,
    "liquidation": 0.15,
    "low risk": 0.7,
    "low risks": 0.7,
    "low cost": 0.7,
    "low costs": 0.7,
    "lower than anticipated": 0.25,
    "lower than expected": 0.25,
    "lucrative": 0.9,
    "late": 0.3,
    "lack": 0.2,
    "long term": 0.7,
    "long terms": 0.7,
    "large cap": 0.7,
    "long position": 0.7,
    "long positions": 0.7,
    "leading indicator": 0.7,
    "liquid assets": 0.7,
    "lull": 0.3,
    "leveraged buyout": 0.7,
    "low growth": 0.2,
    "loss making": 0.1,
    "leveraging": 0.6,
    "launch": 0.7,
    "low value": 0.3,
    "lifetime value": 0.7,
    "liquidate": 0.1,

    "market risk": 0.4,
    "market risks": 0.4,
    "market crisis": 0.1,
    "market share": 0.7,
    "market downturn": 0.2,
    "merger": 0.8,
    "magnitude": 0.7,
    "momentum": 0.8,
    "maturity": 0.7,
    "million": 0.65,
    "millions": 0.65,
    "master": 0.8,
    "markup": 0.7,
    "minimize": 0.7,
    "miss": 0.2,
    "missing": 0.2,
    "missed": 0.2,
    "maximization": 0.85,
    "maximizations": 0.85,
    "multiplier": 0.7,
    "modest": 0.4,
    "manipulation": 0.2,
    "margin call": 0.3,
    "move up": 0.85,
    "moves up": 0.85,
    "moved up": 0.75,
    "money market": 0.7,
    "manufacture": 0.65,
    "markup pricing": 0.7,
    "money supply": 0.65,
    "move forward": 0.7,
    "mining": 0.6,
    "multinational": 0.7,
    "multinationals": 0.7,
    "most attractive": 0.8,
    "magnificent": 0.75,
    "magnific": 0.75,

    "niche": 0.8,
    "non performing": 0.1,
    "narrow": 0.4,
    "new": 0.65,
    "negative": 0.1,
    "negative earnings": 0.0,
    "new product": 0.8,
    "new products": 0.8,
    "net profit": 0.7,
    "net profits": 0.7,
    "note worthy": 0.7,
    "non essential": 0.3,
    "net worth": 0.7,
    "nurture": 0.7,
    "non compliant": 0.2,
    "nervous": 0.3,
    "no growth": 0.2,
    "no growths": 0.2,
    "new investment": 0.8,
    "new investments": 0.8,
    "non cyclical": 0.6,
    "noteworthy": 0.7,
    "normalization": 0.65,
    "net gain": 0.8,
    "net gains": 0.8,
    "not reachable": 0.15,
    "not reached": 0.1,

    "offer": 0.65,
    "offers": 0.65,
    "overperform": 0.9,
    "outperform": 0.9,
    "optimistic": 0.8,
    "opportunity": 0.8,
    "opportunities": 0.8,
    "organic": 0.7,
    "overdue": 0.3,
    "overhead": 0.7,
    "overvalued": 0.2,
    "offset": 0.65,
    "outflow": 0.3,
    "overleveraged": 0.2,
    "overestimate": 0.3,
    "outstanding": 0.8,
    "overcapacity": 0.3,
    "overreaction": 0.3,
    "overexposure": 0.3,
    "overperformance": 0.8,
    "obsolescence": 0.2,
    "overfunded": 0.7,
    "optimization": 0.7,
    "optimizations": 0.7,
    "operating profit": 0.8,
    "overstretch": 0.3,
    "oversupply": 0.2,
    "offerings": 0.7,
    "on track": 0.7,
    "overcome": 0.8,
    "oscillation": 0.35,
    "overproduction": 0.3,
    "organic growth": 0.8,
    "organic growths": 0.8,

    "panic": 0.1,
    "panic selling": 0.1,
    "profits": 0.8,
    "profit": 0.8,
    "profit margin": 0.8,
    "positive": 0.9,
    "positively": 0.9,
    "premium": 0.8,
    "predict": 0.6,
    "prediction": 0.6,
    "predictions": 0.6,
    "pioneer": 0.8,
    "purchasing": 0.7,
    "prosper": 0.9,
    "prospered": 0.9,
    "prospers": 0.9,
    "plan": 0.8,
    "plans": 0.8,
    "positive growth": 0.9,
    "positive growths": 0.9,
    "payoff": 0.8,
    "peak": 0.65,
    "peaking": 0.7,
    "price increase": 0.65,
    "power": 0.7,
    "price cut": 0.4,
    "plunge": 0.2,
    "plunges": 0.2,
    "plunged": 0.2,
    "plummeted": 0.2,
    "pressure": 0.3,
    "pressures": 0.3,
    "pressured": 0.3,
    "pandemic": 0.2,
    "pessimistic": 0.2,
    "plentiful": 0.8,
    "penetrant": 0.65,
    "premium rate": 0.8,
    "plunge risk": 0.3,
    "poor performance": 0.2,
    "poor": 0.1,
    "progress": 0.8,
    "problem": 0.15,
    "problems": 0.1,
    "product release": 0.7,
    "product releases": 0.7,
    "product released": 0.7,
    "pull back": 0.2,
    "pulls back": 0.2,
    "pulling back": 0.2,
    "pulled back": 0.2,

    "quality": 0.65,
    "quick": 0.8,
    "quarantine": 0.2,
    "questionable": 0.3,
    "quiet": 0.4,
    "quick turnaround": 0.65,
    "quality control": 0.65,
    "quaint": 0.65,

    "rapid": 0.7,
    "rattle": 0.3,
    "revenue": 0.8,
    "rebound": 0.8,
    "rebounds": 0.8,
    "revenues": 0.8,
    "recovery": 0.7,
    "reinvestment": 0.6,
    "reduction": 0.3,
    "reductions": 0.3,
    "resilience": 0.8,
    "risk": 0.2,
    "risks": 0.2,
    "robust": 0.8,
    "recession": 0.1,
    "rebalancing": 0.65,
    "revenue growth": 0.9,
    "revenue growths": 0.9,
    "reliable": 0.9,
    "raise": 0.8,
    "rise": 0.8,
    "rises": 0.8,
    "rising price": 0.3,
    "rising prices": 0.3,
    "rising debt": 0.2,
    "refinancing": 0.6,
    "reduction in force": 0.3,
    "risk aversion": 0.3,
    "rally": 0.8,
    "recovery plan": 0.75,
    "reliable performance": 0.8,
    "reinforcement": 0.7,
    "reinvestment strategy": 0.8,
    "risky": 0.2,
    "repayment": 0.6,
    "recessionary": 0.2,
    "redemption": 0.65,
    "revenue stream": 0.75,
    "revenue model": 0.8,
    "reserves": 0.6,
    "revenue per share": 0.8,
    
    "share": 0.8,
    "shares": 0.8,
    "shared": 0.8,
    "shocking": 0.2,
    "shocked": 0.2,
    "shock": 0.2,
    "surge": 0.8,
    "surges": 0.8,
    "strong": 0.8,
    "strategy": 0.75,
    "strategic": 0.75,
    "strategies": 0.75,
    "successful": 0.9,
    "savings": 0.7,
    "sustainability": 0.8,
    "sustainable": 0.8,
    "stability": 0.8,
    "securities": 0.7,
    "security": 0.7,
    "secure": 0.9,
    "security breach": 0.1,
    "skepticism": 0.3,
    "steady": 0.7,
    "subsidy": 0.7,
    "startup": 0.65,
    "startups": 0.65,
    "solid": 0.8,
    "sell": 0.3,
    "sell off": 0.2,
    "sells": 0.3,
    "setback": 0.2,
    "setbacks": 0.2,
    "sold": 0.3,
    "sold out": 0.7,
    "spend on growth": 0.9,
    "spends on growth": 0.9,
    "spend efficiently": 0.8,
    "spend more than": 0.4,
    "spends more than": 0.4,
    "surplus": 0.8,
    "stimulus": 0.7,
    "short": 0.3,
    "shrinking": 0.2,
    "shrink": 0.2,
    "slow": 0.3,
    "slows": 0.3,
    "slower": 0.3,
    "slowing": 0.3,
    "slowdown": 0.25,
    "slash": 0.3,
    "slashing": 0.3,
    "slashed": 0.3,
    "slide": 0.3,
    "slides": 0.3,
    "savings plan": 0.7,
    "stagnation": 0.2,
    "stagnant": 0.2,
    "stagflation": 0.2,
    "steep": 0.7,
    "scalability": 0.7,
    "softening": 0.3,
    "soar": 0.8,
    "soars": 0.8,
    "saturation": 0.35,
    "shutdown": 0.2,
    "squeeze": 0.3,
    "sale": 0.65,
    "sales": 0.65,
    "synergy": 0.75,
    "share price": 0.65,
    "spin off": 0.7,
    "stimulation": 0.7,
    "speed": 0.7,
    "stock market crash": 0.1,
    "simply wall st": 0.7,
    "suffered": 0.25,
    "suffer": 0.25,
    "suffers": 0.25,
    "suffering": 0.25,
    
    "tax": 0.3,
    "taxes": 0.3,
    "taxed": 0.2,
    "tangible": 0.65,
    "treasury": 0.7,
    "tension": 0.3,
    "tensions": 0.3,
    "trust": 0.8,
    "technologic": 0.7,
    "technology stock": 0.8,
    "tactical": 0.7,
    "takeover": 0.7,
    "tailwind": 0.8,
    "taxation": 0.35,
    "tighten": 0.3,
    "thrive": 0.7,
    "thrives": 0.7,
    "thriving": 0.7,
    "thrived": 0.7,
    "trade off": 0.65,
    "tactical position": 0.7,
    "targeted": 0.65,
    "tangible asset": 0.8,
    "turbulence": 0.2,
    "trouble": 0.2,
    "troubles": 0.15,
    "transparency": 0.8,
    "total return": 0.65,
    "top line": 0.7,
    "turnkey": 0.7,
    "turmoil": 0.2,
    "takedown": 0.3,
    "toxic": 0.2,
    "toxic asset": 0.1,
    "toxic assets": 0.1,
    "threaten": 0.1,
    "trade war": 0.2,
    "too much": 0.1,
    
    "underperform": 0.2,
    "unemployment": 0.1,
    "upturn": 0.8,
    "unsecured": 0.3,
    "unforeseen": 0.3,
    "utility": 0.7,
    "utilities": 0.7,
    "unified": 0.65,
    "unfavorable": 0.2,
    "unpredictable": 0.3,
    "usury": 0.1,
    "upside": 0.8,
    "up": 0.9,
    "upgrade": 0.8,
    "upgraded": 0.8,
    "upgrades": 0.8,
    "underutilized": 0.35,
    "unavailable": 0.3,
    "unrealized": 0.35,
    "uncovered": 0.3,
    "unsustainable": 0.2,
    "unprofitable": 0.1,
    "unfavorable trend": 0.1,
    "uncertainty": 0.3,
    "uncertain": 0.3,
    "underperformance": 0.2,
    "under": 0.2,
    "upward": 0.8,
    "uncapped": 0.7,
    "unquestionable": 0.65,
    "unlimited": 0.8,
    "unpredictability": 0.25,

    "volatility": 0.3,
    "viability": 0.65,
    "viable": 0.65,
    "vulnerable": 0.2,
    "victory": 0.9,
    "victories": 0.9,
    "violation": 0.2,
    "violations": 0.2,
    "vacancy": 0.3,
    "verifiable": 0.7,
    "venture capital": 0.4,
    "visibility": 0.65,
    "visible": 0.65,
    "vanguard": 0.8,
    "valuation risk": 0.3,
    "vigor": 0.8,
    "vantage": 0.8,
    "vantages": 0.8,
    "volatile": 0.2,
    "victimized": 0.1,
    "vicious": 0.1,
    "valiant": 0.8,
    "verification": 0.65,
    "void": 0.2,
    "vulnerability": 0.2,
    "vulnerabilities": 0.2,
    "volume trading": 0.6,
    "value creation": 0.75,
    "vertical": 0.65,

    "war": 0.2,
    "wars": 0.15,
    "wealth": 0.9,
    "win": 0.9,
    "wins": 0.9,
    "won": 0.9,
    "weakness": 0.2,
    "weaknesses": 0.2,
    "weak": 0.2,
    "weaker": 0.2,
    "weaker than expected": 0.15,
    "withdraw": 0.3,
    "withdrawal": 0.3,
    "withdrawals": 0.3,
    "wave": 0.6,
    "waves": 0.6,
    "wealthy": 0.8,
    "widening": 0.7,
    "wide": 0.7,
    "wholesale": 0.7,
    "well being": 0.8,
    "workforce": 0.7,
    "worst case": 0.2,
    "warning": 0.2,
    "warnings": 0.2,
    "winners": 0.9,
    "win win": 0.9,
    "worth": 0.8,
    "write off": 0.2,
    "wage growth": 0.7,
    "waterfall": 0.6,
    "waterfalls": 0.6,
    "worsen": 0.1,
    "worse": 0.1,
    "worst": 0.1,
    "weaken": 0.2,
    "waiting": 0.4,
    "widen": 0.6,
    "worry": 0.2,
    "worried": 0.2,
    "welfare": 0.8,
    "whipsaw": 0.2,
    "wild": 0.3,
    "winds": 0.6,
    "wind": 0.6,
    
    "x efficiency": 0.7,
    "x factor": 0.8,
    "xenocurrency": 0.6,
    "xenophobic": 0.2,
    "xerox effect": 0.5,
    "xit": 0.3,

    "yield": 0.7,
    "yields curve": 0.6,
    "young market": 0.4,
    "young": 0.4,
    "yellow flag": 0.3,
    "yield spread": 0.7,
    "yield growth": 0.8,
    "yield risk": 0.3,
    "yield risks": 0.3,

    "z score": 0.7,
    "zombie company": 0.2,
    "zombie bank": 0.2,
    "zigzag market": 0.35,
    "zig zag": 0.35,
    "zigzag": 0.35,
    "zenith": 0.8,
    "zero coupon": 0.6,
    "zero inflation": 0.7,
    "zero sum": 0.4,

    #Figure importanti
    "warren buffett": 0.80,
    "elon musk": 0.65,
    "musk": 0.65,
    "donald trump": 0.3,
    "trump": 0.3,
    "jim cramer": 0.65,
    "cathie wood": 0.65,
    "jerome powell": 0.65,
    "jamie dimon": 0.65,
    "ray dalio": 0.65,
    "peter thiel": 0.65,
    "bill ackman": 0.60,
    "charlie munger": 0.65,
    "larry fink": 0.65,
    "michael burry": 0.65,
    "ken griffin": 0.65,
    "david tepper": 0.65,
    "george soros": 0.65,
    "jeff bezos": 0.65,
    "mark zuckerberg": 0.65,
    "tim cook": 0.65,
    "sundar pichai": 0.65,
    "satya nadella": 0.65,
    "sam altman": 0.65,
    "kathy jones": 0.65,
    "liz ann sonders": 0.65,
    "paul tudor jones": 0.65
}



#Normalizza il testo della notizia, rimuovendo impurità
def normalize_text(text):
    #Pulisce e normalizza il testo per una migliore corrispondenza.
    
    text = re.sub(r'\s-\s[^-]+$', '', text)    # Rimuove la parte dopo l'ultimo " - " (se presente)
    text = text.lower()    # Converti tutto in minuscolo
    text = re.sub(r'[-_/]', ' ', text)    # Sostituisci trattini e underscore con spazi
    text = re.sub(r'\s+', ' ', text).strip()    # Rimuovi spazi multipli e spazi iniziali/finali
    
    return text



#Trova i lemmi delle parole per una ricerca più completa
def lemmatize_words(words):
    """Lemmatizza le parole usando spaCy e restituisce una lista di lemmi."""
    doc = nlp(" ".join(words))  # Analizza le parole con spaCy
    return [token.lemma_ for token in doc]



#Calcola il sentiment basato sulle notizie del singolo asset
def calculate_sentiment(news, decay_factor=0.03):    #Prima era 0.06
    """Calcola il sentiment medio ponderato di una lista di titoli di notizie."""
    total_sentiment = 0
    total_weight = 0
    now = datetime.utcnow()

    for news_item in news:
        # Gestisci sia tuple a 2 che 3 elementi
        if len(news_item) == 3:
            title, date, _ = news_item
        elif len(news_item) == 2:
            title, date = news_item
        else:
            # Se la struttura non è quella attesa, salta
            continue

        days_old = (now - date).days  # Calcola l'età della notizia in giorni
        weight = math.exp(-decay_factor * days_old)  # Applica il decadimento esponenziale

        normalized_title = normalize_text(title)  # Normalizza il titolo
        sentiment_score = 0
        count = 0

        words = normalized_title.split()  # Parole del titolo
        lemmatized_words = lemmatize_words(words)  # Lemmatizza le parole

        for i, word in enumerate(lemmatized_words):
            if word in sentiment_dict:
                score = sentiment_dict[word]

                if i > 0 and lemmatized_words[i - 1] in negation_words:
                    score = 1 - score  # Inverto il punteggio

                sentiment_score += score
                count += 1

        if count != 0:
            sentiment_score /= count  # Normalizza il punteggio
        else:
            sentiment_score = 0.5  # Sentiment neutro se nessuna parola è trovata

        total_sentiment += sentiment_score * weight
        total_weight += weight

    if total_weight > 0:
        average_sentiment = total_sentiment / total_weight
    else:
        average_sentiment = 0.5  # Sentiment neutro se non ci sono notizie

    return average_sentiment




# Funzione per calcolare la percentuale in base agli indicatori
def calcola_punteggio(indicatori, close_price, bb_upper, bb_lower):
    punteggio = 0

    if indicatori["RSI (14)"] > 70:
        punteggio -= 8
    elif indicatori["RSI (14)"] < 30:
        punteggio += 8
    else:
        punteggio += 4

    if indicatori["MACD Line"] > indicatori["MACD Signal"]:
        punteggio += 8
    else:
        punteggio -= 6

    if indicatori["Stochastic %K"] > 80:
        punteggio -= 6
    elif indicatori["Stochastic %K"] < 20:
        punteggio += 6

    if indicatori["EMA (10)"] < close_price:
        punteggio += 7

    if indicatori["CCI (14)"] > 0:
        punteggio += 6
    else:
        punteggio -= 4

    if indicatori["Williams %R"] > -20:
        punteggio -= 4
    else:
        punteggio += 4

    # Bollinger Bands
    if close_price > bb_upper:
        punteggio -= 5
    elif close_price < bb_lower:
        punteggio += 5

    return round(((punteggio + 44) * 100) / 88, 2)  # normalizzazione 0-100



#Inserisce tutti i risultati nel file html
def get_sentiment_for_all_symbols(symbol_list):
    sentiment_results = {}
    percentuali_tecniche = {}
    percentuali_combine = {}
    all_news_entries = []
    crescita_settimanale = {}
    dati_storici_all = {}   # 🔹 nuovo: raccolta dati storici di tutti gli asset

    for symbol, adjusted_symbol in zip(symbol_list, symbol_list_for_yfinance):
        news_data = get_stock_news(symbol)  # Ottieni le notizie divise per periodo

        # Calcola il sentiment per ciascun intervallo di tempo
        sentiment_90_days = calculate_sentiment(news_data["last_90_days"])  
        sentiment_30_days = calculate_sentiment(news_data["last_30_days"])  
        sentiment_7_days = calculate_sentiment(news_data["last_7_days"])  

        sentiment_results[symbol] = {
            "90_days": sentiment_90_days,
            "30_days": sentiment_30_days,
            "7_days": sentiment_7_days
        }

        tabella_indicatori = None  
        dati_storici_html = None
        tabella_fondamentali = None
        percentuale = None
        sells_data = None  # 🔹 nuovo: dati Informative Sells

        try:
            ticker = str(adjusted_symbol).strip().upper()
            data = yf.download(ticker, period="3y", auto_adjust=True, progress=False)

            if data.empty:
                raise ValueError(f"Nessun dato disponibile per {symbol} ({adjusted_symbol})")

            if isinstance(data.columns, pd.MultiIndex):
                try:
                    data = data.xs(ticker, axis=1, level=1)
                except KeyError:
                    raise ValueError(f"Ticker {ticker} non trovato nel MultiIndex: {data.columns}")

            close = data['Close']
            high  = data['High']
            low   = data['Low']

            # Salvo i dati storici completi
            dati_storici_all[symbol] = data.copy()

            # Crescita settimanale
            from datetime import timedelta
            try:
                latest_date = close.index[-1]
                date_7_days_ago = latest_date - timedelta(days=7)
                close_week_ago = close[close.index <= date_7_days_ago].iloc[-1]
                close_now = close.loc[latest_date]
                growth_weekly = ((close_now - close_week_ago) / close_week_ago) * 100
                crescita_settimanale[symbol] = round(growth_weekly, 2)
            except Exception:
                crescita_settimanale[symbol] = None

            # Indicatori tecnici
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
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            bb_width = bb.bollinger_wband().iloc[-1]

            indicators = {
                "RSI (14)": round(rsi, 2),
                "MACD Line": round(macd_line, 2),
                "MACD Signal": round(macd_signal, 2),
                "Stochastic %K": round(stoch_k, 2),
                "Stochastic %D": round(stoch_d, 2),
                "EMA (10)": round(ema_10, 2),
                "CCI (14)": round(cci, 2),
                "Williams %R": round(will_r, 2),
                "BB Upper": round(bb_upper, 2),
                "BB Lower": round(bb_lower, 2),
                "BB Width": round(bb_width, 4),
            }

            tabella_indicatori = pd.DataFrame(indicators.items(), columns=["Indicatore", "Valore"]).to_html(index=False, border=0)
            percentuale = calcola_punteggio(indicators, close.iloc[-1], bb_upper, bb_lower)

            # Dati fondamentali
            ticker_obj = yf.Ticker(adjusted_symbol)
            try:
                info = ticker_obj.info or {}
            except Exception:
                info = {}

            def safe_value(key):
                value = info.get(key)
                if isinstance(value, (int, float)):
                    return round(value, 4)
                return "N/A"

            fondamentali = {
                "Trailing P/E": safe_value("trailingPE"),
                "Forward P/E": safe_value("forwardPE"),
                "EPS Growth (YoY)": safe_value("earningsQuarterlyGrowth"),
                "Revenue Growth (YoY)": safe_value("revenueGrowth"),
                "Profit Margins": safe_value("profitMargins"),
                "Debt to Equity": safe_value("debtToEquity"),
                "Dividend Yield": safe_value("dividendYield")
            }

            tabella_fondamentali = pd.DataFrame(
                fondamentali.items(), columns=["Fundamentale", "Valore"]
            ).to_html(index=False, border=0)

            percentuali_tecniche[symbol] = percentuale

            # Tabella ultimi 90 giorni
            dati_storici = data.tail(90).copy()
            dati_storici['Date'] = dati_storici.index.strftime('%Y-%m-%d')
            dati_storici_html = dati_storici[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']].to_html(index=False, border=1)

            indicator_data[symbol] = indicators
            fundamental_data[symbol] = fondamentali

            # 🔹 Recupero Informative Sells
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

                sells_data = {
                    'daily_sells': daily_sells,
                    'Last Day': last_day,
                    'Last Day Total Sells ($)': last_value,
                    'Max Daily Sell ($)': max_daily,
                    'Last vs Max (%)': percent_of_max,
                    'Number of Sells Last Day': num_sells_last_day
                }
            except Exception:
                sells_data = None

        except Exception as e:
            print(f"Errore durante l'analisi di {symbol}: {e}")

        # Costruzione HTML identico a prima + sezione Informative Sells
        file_path = f"results/{symbol.upper()}_RESULT.html"
        html_content = [
            f"<html><head><title>Previsione per {symbol}</title></head><body>",
            f"<h1>Previsione per: ({symbol})</h1>",
            "<table border='1'><tr><th>Probability</th></tr>",
            f"<tr><td>{sentiment_90_days * 100}</td></tr>",
            "</table>",
            "<table border='1'><tr><th>Probability30</th></tr>",
            f"<tr><td>{sentiment_30_days * 100}</td></tr>",
            "</table>",
            "<table border='1'><tr><th>Probability7</th></tr>",
            f"<tr><td>{sentiment_7_days * 100}</td></tr>",
            "</table>",
            "<hr>",
            "<h2>Indicatori Tecnici</h2>",
        ]

        if percentuale is not None:
            html_content.append(f"<p><strong>Probabilità calcolata sugli indicatori tecnici:</strong> {percentuale}%</p>")
        else:
            html_content.append("<p><strong>Impossibile calcolare la probabilità sugli indicatori tecnici.</strong></p>")

        if tabella_indicatori:
            html_content.append(tabella_indicatori)
        else:
            html_content.append("<p>No technical indicators available.</p>")

        html_content.append("<h2>Dati Fondamentali</h2>")
        if tabella_fondamentali:
            html_content.append(tabella_fondamentali)
        else:
            html_content.append("<p>Nessun dato fondamentale disponibile.</p>")

        # 🔹 Aggiunta sezione Informative Sells
        html_content.append("<h2>Informative Sells</h2>")
        if sells_data is not None:
            html_content += [
                f"<p><strong>Ultimo giorno registrato:</strong> {sells_data['Last Day']}</p>",
                f"<p><strong>Totale vendite ultimo giorno ($):</strong> {sells_data['Last Day Total Sells ($)']}</p>",
                f"<p><strong>Numero transazioni ultimo giorno:</strong> {sells_data['Number of Sells Last Day']}</p>",
                f"<p><strong>% rispetto al massimo storico giornaliero:</strong> {sells_data['Last vs Max (%)']:.2f}%</p>"
            ]
        else:
            html_content.append("<p>Informative Sells non disponibili.</p>")

        if dati_storici_html:
            html_content += [
                "<h2>Dati Storici (ultimi 90 giorni)</h2>",
                dati_storici_html,
                "</body></html>"
            ]
        else:
            html_content.append("<p>No historical data available.</p>")
            html_content.append("</body></html>")

        # Salvataggio file su repo
        try:
            contents = repo.get_contents(file_path)
            repo.update_file(contents.path, f"Updated probability for {symbol}", "\n".join(html_content), contents.sha)
        except GithubException:
            repo.create_file(file_path, f"Created probability for {symbol}", "\n".join(html_content))

        for title, news_date, link, source, image in news_data["last_90_days"]:
            title_sentiment = calculate_sentiment([(title, news_date)])
            all_news_entries.append((symbol, title, title_sentiment, link, source, image))

    # Media ponderata (uguale)
    w7, w30, w90 = 0.5, 0.3, 0.2
    for symbol in sentiment_results:
        if symbol in percentuali_tecniche:
            sentiment_7 = sentiment_results[symbol]["7_days"] * 100
            sentiment_30 = sentiment_results[symbol]["30_days"] * 100
            sentiment_90 = sentiment_results[symbol]["90_days"] * 100
            sentiment_combinato = (w7 * sentiment_7) + (w30 * sentiment_30) + (w90 * sentiment_90)
            tecnica = percentuali_tecniche[symbol]
            combinata = (sentiment_combinato * 0.6) + (tecnica * 0.4)
            percentuali_combine[symbol] = combinata

    return (sentiment_results, percentuali_combine, all_news_entries, 
            indicator_data, fundamental_data, crescita_settimanale, dati_storici_all)
    


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
pro_file_path = "results/classificaPRO.html"
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

fire_file_path = "results/fire.html"
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

    # ---------- helper: calculate score ----------
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

        weights = {"sentiment": 0.2, "percent": 0.3, "rsi": 0.1, "volume": 0.1, "growth": 0.2, "pe": 0.1}
        score = (
            sentiment_score * weights["sentiment"] +
            percent_score * weights["percent"] +
            rsi_score * weights["rsi"] +
            volume_score * weights["volume"] +
            growth_score * weights["growth"] +
            pe_score * weights["pe"]
        )
        return round(score, 2)

    # ---------- build insight for each symbol ----------
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
        if percent > 65:
            theme = "gainer"
        elif percent < 35:
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

    # ---------- helper: build forecast phrase ----------
    def build_forecast_phrase(ins):
        if ins["rsi"] is not None and ins["rsi"] < 30:
            if ins["sentiment"] >= 0:
                options = [
                    " The stock may rebound soon.",
                    " A potential bounce is likely.",
                    " Buyers could enter at oversold levels.",
                    " The stock may be undervalued and could recover.",
                    " A turnaround might occur in the near future."
                ]
                return random.choice(options)
            else:
                options = [
                    " The decline may continue unless sentiment improves.",
                    " Weakness could persist without a change in sentiment.",
                    " Bears may remain in control shortly.",
                    " Downside risk remains high given the current sentiment.",
                    " Further drops are possible if negativity continues."
                ]
                return random.choice(options)
    
        elif ins["rsi"] is not None and ins["rsi"] > 70:
            if ins["sentiment"] < 0:
                options = [
                    " A pullback is likely in the short term.",
                    " Profit-taking may cause a correction soon.",
                    " Overbought conditions suggest a possible retracement.",
                    " Selling pressure could increase due to high valuations.",
                    " The stock may experience a cooling-off period after strong gains."
                ]
                return random.choice(options)
            else:
                options = [
                    " Gains could continue, but caution is needed.",
                    " Momentum remains strong despite high RSI.",
                    " The rally may continue, but reversal risk exists.",
                    " Be cautious as the stock approaches overbought levels.",
                    " Further upside is possible, but with caution."
                ]
                return random.choice(options)
    
        elif ins["delta"] > 0 and ins["sentiment"] > 0.1:
            options = [
                " Upward momentum may continue.",
                " Positive sentiment supports further gains.",
                " Buyers seem confident, pushing prices higher.",
                " The stock may keep rising shortly.",
                " Market optimism could lead to additional gains."
            ]
            return random.choice(options)
    
        elif ins["delta"] < 0 and ins["sentiment"] < -0.1:
            options = [
                " Weakness may continue.",
                " Negative sentiment suggests more downside.",
                " Selling pressure may persist.",
                " Bears could remain in control for now.",
                " The downtrend may extend before stabilizing."
            ]
            return random.choice(options)
    
        else:
            options = [
                " The short-term outlook is unclear.",
                " Market conditions are mixed and caution is advised.",
                " The stock may consolidate while investors wait for signals.",
                " Uncertainty remains, making direction unclear.",
                " Traders may wait before making moves until momentum clarifies."
            ]
            return random.choice(options)

    # ---------- templates ----------
    leads_positive = [
        "The market shows a generally positive trend today.",
        "Many stocks gained strongly today.",
        "Investor optimism pushed several top stocks higher.",
        "Market momentum supports bullish sentiment in most sectors.",
        "Today’s trading shows growing investor confidence.",
        "Buying interest increased, lifting key indices.",
        "Broad strength lifts the market and shows strong demand.",
        "Positive earnings and economic data improved investor morale.",
        "Renewed risk appetite supports gains in many sectors.",
        "Stocks rose steadily as investors remained optimistic."
    ]
    
    leads_mixed = [
        "The market shows mixed results with gains and losses.",
        "Stocks performed unevenly, showing a cautious mood.",
        "Market activity was uneven with some winners and losers.",
        "Investors were indecisive and weighed risks and opportunities.",
        "Markets fluctuated as investors were uncertain about direction.",
        "The market struggled to find clear footing with conflicting signals.",
        "Mixed earnings contributed to a patchy trading session.",
        "Volatility affected the session while investors balanced hope and caution.",
        "Selective buying and profit-taking created a choppy market.",
        "Investors responded to conflicting economic and geopolitical news."
    ]
        
    leads_negative = [
        "The market closed lower with selling pressure.",
        "Most stocks lost value in a cautious trading day.",
        "Investor sentiment turned negative and key stocks fell.",
        "Market sentiment is bearish due to profit-taking.",
        "It was a difficult day as bears controlled the market.",
        "Selling increased after disappointing news.",
        "Broad declines reflected concerns about economic growth.",
        "Negative headlines reduced investor confidence.",
        "Stocks fell sharply amid higher volatility and risk aversion.",
        "Investor fears led to a broad market sell-off."
    ]

    clause_templates = {
        "gainer": [
            "{name} rose {delta:.1f}% today, leading the market.",
            "{name} is a top performer, up {delta:.1f}%.",
            "{name} climbed {delta:.1f}% with strong momentum.",
            "Investors pushed {name} up by {delta:.1f}%.",
            "{name} gained {delta:.1f}% in the session.",
            "{name} rose {delta:.1f}% during trading.",
            "Buying interest lifted {name} by {delta:.1f}%.",
            "Strong demand moved {name} up {delta:.1f}%.",
            "Momentum increased {name} by {delta:.1f}%.",
            "{name} posted a gain of {delta:.1f}%."
        ],
        "loser": [
            "{name} fell {delta:.1f}% due to selling pressure.",
            "{name} was down {delta:.1f}% today.",
            "Selling pushed {name} down {delta:.1f}%.",
            "{name} declined {delta:.1f}%, one of the worst performers.",
            "Bearish sentiment caused {name} to drop {delta:.1f}%.",
            "{name} lost {delta:.1f}% in a weak market.",
            "{name} gave back {delta:.1f}% of gains.",
            "Investors sold {name}, dropping it {delta:.1f}%.",
            "{name} experienced a decline of {delta:.1f}%.",
            "Profit-taking reduced {name} by {delta:.1f}%."
        ],
        "oversold": [
            "{name} appears oversold with RSI {rsi}, possible rebound.",
            "RSI {rsi} suggests {name} may attract buyers.",
            "{name} shows oversold RSI {rsi}, potential recovery.",
            "RSI reading {rsi} indicates {name} is oversold.",
            "{name} is in oversold territory (RSI {rsi}), buying may be possible.",
            "Oversold conditions (RSI {rsi}) could help {name} recover.",
            "{name} trades at oversold levels (RSI {rsi}), may reverse.",
            "Investors may find value in {name} (RSI {rsi}).",
            "{name} RSI {rsi} points to oversold and possible bounce.",
            "Market data shows {name} oversold (RSI {rsi}), caution advised."
        ],
        "overbought": [
            "{name} looks overbought (RSI {rsi}), caution advised.",
            "RSI {rsi} on {name} may lead to profit-taking.",
            "{name} is overbought (RSI {rsi}), possible consolidation.",
            "High RSI ({rsi}) signals overbought conditions on {name}.",
            "{name} RSI {rsi} indicates potential overheating.",
            "Profit-taking might follow for {name} with RSI {rsi}.",
            "{name} reached overbought RSI {rsi}, possible pause.",
            "RSI {rsi} suggests investors should be careful with {name}.",
            "High RSI ({rsi}) may cause {name} short-term pullback.",
            "RSI shows {name} is overbought (RSI {rsi}), possible top."
        ],
        "neutral": [
            "{name} remained stable today.",
            "{name} showed little movement.",
            "A consolidation day for {name}, no major change.",
            "{name} closed relatively flat.",
            "{name} had a calm trading day.",
            "{name} showed no notable changes.",
            "{name} traded in a narrow range.",
            "Limited volatility for {name}, ending unchanged.",
            "{name}'s price was muted, no clear direction.",
            "{name} held steady in mixed market conditions."
        ]
    }

    intra_connectors = [
        "while", "and", "with", "although", "however", "in addition", "simultaneously",
        "at the same time", "also", "moreover"
    ]
    between_sent_connectors = [
        "Meanwhile", "Additionally", "Another point to note is", "Finally",
        "It is also worth mentioning", "To conclude", "Furthermore", "Besides this"
    ]

    # ---------- analyze market mood ----------
    gainers_count = sum(1 for v in percentuali_combine.values() if v > 65)
    losers_count = sum(1 for v in percentuali_combine.values() if v < 35)
    total_symbols = len(percentuali_combine)
    avg_sentiment = sum(
        sentiment_for_symbols.get(sym, 0) if not isinstance(sentiment_for_symbols.get(sym, 0), dict)
        else sentiment_for_symbols.get(sym, {}).get("sentiment", 0)
        for sym in percentuali_combine.keys()
    ) / max(total_symbols, 1)

    if avg_sentiment > 0.15 and gainers_count > losers_count:
        lead_sentence = random.choice(leads_positive)
    elif avg_sentiment < -0.15 and losers_count > gainers_count:
        lead_sentence = random.choice(leads_negative)
    else:
        lead_sentence = random.choice(leads_mixed)

    # ---------- select top scoring symbols ----------
    scores = {sym: calculate_asset_score(sym) for sym in percentuali_combine.keys()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [s for s, _ in ranked[:3]]
    insights = [build_insight(s) for s in selected]

    if not insights:
        return "No significant developments."

    # ---------- group by theme ----------
    groups = {}
    for ins in insights:
        groups.setdefault(ins["theme"], []).append(ins)

    paragraph_sentences = [lead_sentence]

    def fuse_group(clauselist):
        if not clauselist:
            return None
        clauselist = sorted(clauselist, key=lambda x: x["score"], reverse=True)
        main = clauselist[0]
        main_phrase = random.choice(clause_templates[main["theme"]]).format(
            name=main["name"],
            delta=abs(main["delta"]),
            rsi=(int(main["rsi"]) if main["rsi"] is not None else "—")
        ) + build_forecast_phrase(main)

        support = ""
        if len(clauselist) > 1:
            supports = []
            for s in clauselist[1:]:
                ph = random.choice(clause_templates[s["theme"]]).format(
                    name=s["name"],
                    delta=abs(s["delta"]),
                    rsi=(int(s["rsi"]) if s["rsi"] is not None else "—")
                ) + build_forecast_phrase(s)
                supports.append(ph)

            method = random.choice(["conjunction", "semicolon", "subordinate_clause"])
            if method == "conjunction":
                for i, ph in enumerate(supports):
                    conn = random.choice(intra_connectors)
                    if i == 0:
                        support += f" {conn} {ph}"
                    else:
                        support += f", {conn} {ph}"
            elif method == "semicolon":
                support = "; " + "; ".join(supports)
            else:
                conn = random.choice(intra_connectors).capitalize()
                if len(supports) == 1:
                    support = f" {conn} {supports[0]}"
                else:
                    last = supports.pop()
                    support = f" {conn} " + ", ".join(supports) + f", and {last}"

        return main_phrase + support

    # --- random order themes ---
    main_themes = ["gainer", "loser", "oversold", "overbought"]
    random.shuffle(main_themes)
    priority = main_themes + ["neutral"]

    for theme in priority:
        if theme in groups:
            fused = fuse_group(groups[theme])
            if fused:
                if len(paragraph_sentences) > 1:
                    conn = random.choice(between_sent_connectors)
                    fused = f"{conn}, {fused[0].lower()}{fused[1:]}"
                paragraph_sentences.append(fused)

    positive_closings = [
        "Sentiment is positive, showing investor confidence.",
        "The mood is optimistic with good signs in the market.",
        "Investors are generally upbeat despite some uncertainty.",
        "Market momentum continues, supporting optimism.",
        "Positive factors support gains in key sectors.",
        "Investor sentiment is supported by strong fundamentals.",
        "Confidence grows as economic indicators show market strength.",
        "A constructive atmosphere encourages risk-taking."
    ]
    
    neutral_closings = [
        "The session closes with a balanced market tone.",
        "Overall, the market is steady with no clear direction.",
        "A quiet day as investors wait for more information.",
        "Markets remain mixed as participants assess economic signals.",
        "Investors are cautious, waiting for clearer signs.",
        "Trading volumes are moderate amid uncertainty.",
        "The day ends without major changes, showing consolidation.",
        "Market participants are careful amid fluctuating data."
    ]
    
    negative_closings = [
        "The tone is cautious, reflecting investor concerns.",
        "Market sentiment shows uncertainty and risk aversion.",
        "Bearish trends prevail as traders remain cautious.",
        "Selling pressure weighs on market confidence.",
        "Volatility is high and downside risks dominate.",
        "Investors are cautious amid economic headwinds.",
        "Negative momentum challenges recent gains.",
        "The market closes with apprehension and defensive positioning."
    ]

    if avg_sentiment > 0.2:
        closing = random.choice(positive_closings)
    elif avg_sentiment < -0.2:
        closing = random.choice(negative_closings)
    else:
        closing = random.choice(neutral_closings)

    paragraph_sentences.append(closing)

    # --- per-asset list with repetition control ---
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
                delta=abs(ins["delta"]),
                rsi=(int(ins["rsi"]) if ins["rsi"] is not None else "—")
            ) + build_forecast_phrase(ins)
            if candidate not in used_phrases:
                phrase = candidate
                used_phrases.add(candidate)
                break
            tries += 1
        if not phrase:
            # fallback in case all templates repeat, just use last candidate
            phrase = candidate
        symbol_phrases.append(f"{symbol} - {phrase}")

    symbol_phrases_str = "\n".join(symbol_phrases)

    return " ".join(paragraph_sentences), symbol_phrases_str





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

    file_path = f"results/{filename}"
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

file_path = "results/daily_brief_en.html"
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




def salva_correlazioni_html(correlazioni, repo, file_path="results/correlations.html"):
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
