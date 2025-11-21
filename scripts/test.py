import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import os
from email.utils import parsedate_to_datetime

# --- NOVITÀ: LIBRERIE DI ANALISI DEL TESTO ---
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Scarichiamo il dizionario di parole (va fatto una volta sola)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# --- 1. IL MODELLO MATEMATICO (Core Logic) ---
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
        if len(df) < 200: return 0.0
        
        close = df['Close']
        sma_200 = close.rolling(window=200).mean().iloc[-1]
        rsi = self._calculate_rsi(close).iloc[-1]
        current_price = close.iloc[-1]

        score = 0.0
        # Trend
        if current_price > sma_200: score += 0.5
        else: score -= 0.5
        
        # Momentum
        if rsi < 30: score += 0.5
        elif rsi > 70: score -= 0.5
        
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df_history, news_sentiment):
        s_tech = self._get_technical_score(df_history)
        s_news = news_sentiment
        
        # Se c'è una notizia rilevante (positiva o negativa)
        has_news = abs(s_news) > 0.05 

        if has_news:
            # SCENARIO: Notizie presenti (guidano il mercato)
            w_news = 0.60 # Aumentato peso news al 60% perché ora sono reali
            w_tech = 0.40
            mode = "IBRIDA (Sentiment Reale)"
        else:
            # SCENARIO: Nessuna notizia (guida il trend)
            w_news = 0.00
            w_tech = 1.00 
            mode = "SOLO TECNICA (No News Rilevanti)"
        
        final_score = (s_news * w_news) + (s_tech * w_tech)
        
        # Clamp finale per sicurezza
        final_score = max(min(final_score, 1.0), -1.0)
        
        probability = 50 + (final_score * 50)

        return {
            'probability': round(probability, 2),
            'mode': mode,
            'details': {'tech': round(s_tech, 2), 'news': round(s_news, 2)}
        }

# --- 2. ANALIZZATORE SENTIMENT (IL "CERVELLO") ---

def analyze_sentiment_text(titles):
    """
    Analizza una lista di titoli e restituisce un punteggio medio da -1 a 1.
    Usa VADER + un dizionario finanziario custom.
    """
    sia = SentimentIntensityAnalyzer()
    
    # POTENZIAMENTO VADER: Aggiungiamo termini finanziari specifici
    # VADER di base non sa che "bullish" è positivo. Glielo insegniamo.
    financial_lexicon = {
        'surge': 4.0, 'jump': 3.5, 'rally': 3.5, 'soar': 4.0, 'bull': 3.0, 'bullish': 3.5,
        'high': 2.0, 'gain': 2.5, 'profit': 3.0, 'beat': 2.5, 'strong': 2.5, 'growth': 2.0,
        'plunge': -4.0, 'crash': -4.0, 'drop': -3.0, 'slump': -3.5, 'bear': -3.0, 'bearish': -3.5,
        'loss': -3.0, 'miss': -2.5, 'weak': -2.5, 'fall': -2.5, 'down': -2.0, 'low': -2.0,
        'inflation': -1.5, 'recession': -3.0, 'crisis': -4.0, 'risk': -1.5, 'cut': -1.5
    }
    sia.lexicon.update(financial_lexicon)
    
    total_score = 0
    count = 0
    
    print(f"     > Analisi del contenuto di {len(titles)} titoli...")
    
    for text in titles:
        # Calcola score del singolo titolo
        score = sia.polarity_scores(text)['compound']
        
        # Stampa di debug per vedere come valuta ogni frase
        # (Utile per capire se sta ragionando bene)
        sentiment_label = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
        print(f"       {sentiment_label} [{score:.2f}] {text[:60]}...")
        
        total_score += score
        count += 1
        
    if count == 0: return 0.0
    
    # Restituisce la media dei sentiment
    return total_score / count

# --- 3. RECUPERO NOTIZIE ---

def get_real_news_analysis(ticker):
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    valid_titles = []
    time_window = timedelta(hours=48)
    now = datetime.now().astimezone() 
    
    try:
        response = requests.get(rss_url, headers=headers, timeout=10)
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        
        print(f"   > Scaricamento news per {ticker}...")
        
        for item in items:
            title = item.find('title').text
            pub_date_str = item.find('pubDate').text
            
            try:
                pub_date = parsedate_to_datetime(pub_date_str)
                age = now - pub_date
                
                if age < time_window:
                    hours_ago = int(age.total_seconds() // 3600)
                    # Aggiungiamo alla lista solo il testo per l'analisi
                    valid_titles.append(title)
            except:
                continue

        if not valid_titles:
            print(f"   > Nessuna notizia recente (48h).")
            return 0.0
        
        # --- QUI AVVIENE LA VALUTAZIONE REALE ---
        avg_sentiment = analyze_sentiment_text(valid_titles)
        print(f"   > Punteggio Sentiment Medio: {avg_sentiment:.4f}")
        return avg_sentiment

    except Exception as e:
        print(f"   [Errore RSS] {e}")
        return 0.0

# --- 4. ESECUZIONE ---

if __name__ == "__main__":
    
    env_tickers = os.environ.get('TICKER_INPUT')
    if env_tickers:
        ASSETS = [t.strip() for t in env_tickers.split(',') if t.strip()]
    else:
        ASSETS = ['AAPL', 'NVDA', 'UCG.MI', 'TSLA', 'BTC-USD']
    
    scorer = HybridScorer()
    
    print(f"\n--- TEST PREVISIONALE REALE (VADER AI) ---")
    
    for ticker in ASSETS:
        print(f"\n📊 Analisi Asset: {ticker}")
        print("-" * 40)
        
        # 1. Dati Storici
        df = get_historical_data(ticker)
        
        # 2. News Reali + Sentiment Analysis
        if not df.empty:
            time.sleep(1)
            sentiment_score = get_real_news_analysis(ticker)
            
            # 3. Calcolo Probabilità
            result = scorer.calculate_probability(df, sentiment_score)
            
            prob = result['probability']
            if prob >= 60: signal = "🟢 STRONG BUY"
            elif prob >= 55: signal = "🟢 BUY"
            elif prob <= 40: signal = "🔴 STRONG SELL"
            elif prob <= 45: signal = "🔴 SELL"
            else: signal = "⚪ HOLD"
            
            print(f"\n   🎯 RISULTATO:")
            print(f"   Segnale: {signal} ({prob}%)")
            print(f"   Modalità: {result['mode']}")
            print(f"   Scores: {result['details']}")
        else:
            print("   [!] Dati insufficienti.")
        
        print("=" * 40)
