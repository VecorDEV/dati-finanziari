import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import os
from email.utils import parsedate_to_datetime

# --- SETUP LIBRERIE AI (NLTK VADER) ---
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Scarica il dizionario necessario (solo la prima volta)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    print("   [Setup] Scaricamento dizionario VADER...")
    nltk.download('vader_lexicon', quiet=True)

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
        # Trend Following
        if current_price > sma_200: score += 0.5
        else: score -= 0.5
        
        # Momentum (RSI)
        if rsi < 30: score += 0.5
        elif rsi > 70: score -= 0.5
        
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df_history, news_sentiment):
        s_tech = self._get_technical_score(df_history)
        s_news = news_sentiment
        
        # Soglia minima per considerare il sentiment "attivo"
        has_news = abs(s_news) > 0.05 

        if has_news:
            # SCENARIO A: Ci sono notizie rilevanti
            # Le news pesano il 60% (perché ora sono analizzate dall'AI)
            w_news = 0.60 
            w_tech = 0.40
            mode = "IBRIDA (Sentiment AI)"
        else:
            # SCENARIO B: Nessuna notizia (o notizie neutre)
            w_news = 0.00
            w_tech = 1.00 
            mode = "SOLO TECNICA (No News Rilevanti)"
        
        final_score = (s_news * w_news) + (s_tech * w_tech)
        
        # Limita il punteggio tra -1 e 1 per sicurezza
        final_score = max(min(final_score, 1.0), -1.0)
        
        # Converte in percentuale (0-100%)
        probability = 50 + (final_score * 50)

        return {
            'probability': round(probability, 2),
            'mode': mode,
            'details': {'tech': round(s_tech, 2), 'news': round(s_news, 2)}
        }

# --- 2. RECUPERO DATI STORICI (Funzione Reinserita) ---

def get_historical_data(ticker):
    """Recupera 1 anno di dati storici per calcolare la media mobile."""
    try:
        stock = yf.Ticker(ticker)
        # Scarica dati storici (senza progress bar)
        df = stock.history(period="1y")
        return df
    except Exception as e:
        print(f"   [Errore Storico] {ticker}: {e}")
        return pd.DataFrame()

# --- 3. ANALISI TESTUALE (Il "Cervello") ---

def analyze_sentiment_text(titles):
    """
    Analizza una lista di titoli usando VADER + Lessico Finanziario.
    """
    sia = SentimentIntensityAnalyzer()
    
    # Dizionario finanziario per migliorare la precisione
    financial_lexicon = {
        'surge': 4.0, 'jump': 3.5, 'rally': 3.5, 'soar': 4.0, 'bull': 3.0, 'bullish': 3.5,
        'high': 2.0, 'gain': 2.5, 'profit': 3.0, 'beat': 2.5, 'strong': 2.5, 'growth': 2.0,
        'plunge': -4.0, 'crash': -4.0, 'drop': -3.0, 'slump': -3.5, 'bear': -3.0, 'bearish': -3.5,
        'loss': -3.0, 'miss': -2.5, 'weak': -2.5, 'fall': -2.5, 'down': -2.0, 'low': -2.0,
        'inflation': -1.5, 'recession': -3.0, 'crisis': -4.0, 'risk': -1.5, 'cut': -1.5,
        'fears': -2.0, 'uncertainty': -1.5, 'warning': -2.0
    }
    sia.lexicon.update(financial_lexicon)
    
    total_score = 0
    count = 0
    
    print(f"     > Analisi AI su {len(titles)} titoli...")
    
    for text in titles:
        score = sia.polarity_scores(text)['compound']
        
        # Stampa di debug per vedere cosa pensa l'AI
        if abs(score) > 0.1: # Mostra solo se non è neutro
            icon = "🟢" if score > 0 else "🔴"
            # Tagliamo il testo se troppo lungo per pulizia log
            print(f"       {icon} [{score:.2f}] {text[:70]}...")
        
        total_score += score
        count += 1
        
    if count == 0: return 0.0
    
    return total_score / count

# --- 4. RECUPERO NEWS RSS ---

def get_real_news_analysis(ticker):
    """Scarica news RSS da Google e le passa all'analizzatore."""
    # Costruisce l'URL di ricerca specifico per il ticker
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    valid_titles = []
    # Finestra temporale di 48 ore
    time_window = timedelta(hours=48)
    now = datetime.now().astimezone()
    
    try:
        print(f"   > Ricerca News RSS per {ticker}...")
        response = requests.get(rss_url, headers=headers, timeout=10)
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        
        for item in items:
            title = item.find('title').text
            pub_date_str = item.find('pubDate').text
            
            try:
                pub_date = parsedate_to_datetime(pub_date_str)
                age = now - pub_date
                
                if age < time_window:
                    valid_titles.append(title)
            except:
                continue

        if not valid_titles:
            print(f"   > Nessuna notizia recente (<48h).")
            return 0.0
        
        # Analisi del sentiment reale
        avg_sentiment = analyze_sentiment_text(valid_titles)
        print(f"   > Sentiment Medio Calcolato: {avg_sentiment:.4f}")
        return avg_sentiment

    except Exception as e:
        print(f"   [Errore RSS] {e}")
        return 0.0

# --- 5. ESECUZIONE ---

if __name__ == "__main__":
    
    # Gestione Input (GitHub Actions o Default Locale)
    env_tickers = os.environ.get('TICKER_INPUT')
    if env_tickers:
        ASSETS = [t.strip() for t in env_tickers.split(',') if t.strip()]
    else:
        # Lista di default per test locale
        ASSETS = ['AAPL', 'NVDA', 'TSLA', 'BTC-USD']
    
    scorer = HybridScorer()
    
    print(f"\n--- TEST PREVISIONALE REALE (VADER AI) ---")
    

    for ticker in ASSETS:
        print(f"\n📊 Analisi Asset: {ticker}")
        print("-" * 40)
        
        # 1. Dati Storici
        df = get_historical_data(ticker)
        
        if not df.empty:
            # Piccolo ritardo per gentilezza verso i server
            time.sleep(1)
            
            # 2. News + Analisi
            sentiment_score = get_real_news_analysis(ticker)
            
            # 3. Calcolo Probabilità
            result = scorer.calculate_probability(df, sentiment_score)
            
            # 4. Output Formattato
            prob = result['probability']
            if prob >= 60: signal = "🟢 STRONG BUY"
            elif prob >= 55: signal = "🟢 BUY"
            elif prob <= 40: signal = "🔴 STRONG SELL"
            elif prob <= 45: signal = "🔴 SELL"
            else: signal = "⚪ HOLD"
            
            print(f"\n   🎯 RISULTATO:")
            print(f"   Segnale: {signal} ({prob}%)")
            print(f"   Modalità: {result['mode']}")
            print(f"   Dettagli: {result['details']}")
        else:
            print("   [!] Dati storici insufficienti o errore download.")
        
        print("=" * 40)
