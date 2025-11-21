import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import random
import os
from email.utils import parsedate_to_datetime # Libreria nativa ottima per date RSS

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
        if current_price > sma_200: score += 0.5
        else: score -= 0.5
        
        if rsi < 30: score += 0.5
        elif rsi > 70: score -= 0.5
        
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df_history, news_sentiment):
        s_tech = self._get_technical_score(df_history)
        s_news = news_sentiment
        
        # Se abbiamo sentiment (diverso da 0), attiviamo la modalità ibrida
        has_news = abs(s_news) > 0.01 

        if has_news:
            w_news = 0.50
            w_tech = 0.50
            mode = "IBRIDA (News + Tech)"
        else:
            w_news = 0.00
            w_tech = 1.00 
            mode = "SOLO TECNICA (No News < 48h)"
        
        final_score = (s_news * w_news) + (s_tech * w_tech)
        probability = 50 + (final_score * 50)

        return {
            'probability': round(probability, 2),
            'mode': mode,
            'details': {'tech': round(s_tech, 2), 'news': round(s_news, 2)}
        }

# --- 2. FUNZIONI DI RECUPERO DATI ---

def get_historical_data(ticker):
    try:
        # Rimuoviamo 'progress' che dava errore
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y") 
        return df
    except Exception as e:
        print(f"   [Errore Storico] {ticker}: {e}")
        return pd.DataFrame()

def get_rss_news_sentiment(ticker):
    """
    Recupera notizie fresche tramite GOOGLE NEWS RSS SEARCH.
    Analizza l'XML e filtra per data.
    """
    # Costruiamo l'URL RSS specifico per il ticker
    # 'q={ticker}+stock' aiuta a focalizzare la ricerca sulla finanza
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    
    valid_news = []
    
    # FINESTRA TEMPORALE: 48 Ore (Ottimale per coprire fusi orari e news recenti)
    time_window = timedelta(hours=48)
    # Bisogna usare timezone-aware datetime perché i feed RSS hanno il fuso orario
    now = datetime.now().astimezone() 

    print(f"   > RSS Search: {rss_url}")
    
    try:
        response = requests.get(rss_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parsing XML
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        
        for item in items:
            title = item.find('title').text
            pub_date_str = item.find('pubDate').text # Es: "Fri, 21 Nov 2025 14:00:00 GMT"
            
            # Parsing della data RSS (gestisce vari formati standard)
            try:
                pub_date = parsedate_to_datetime(pub_date_str)
                
                # Calcolo età
                age = now - pub_date
                
                # FILTRO TEMPORALE
                if age < time_window:
                    valid_news.append({
                        'title': title,
                        'date': pub_date.strftime('%Y-%m-%d %H:%M'),
                        'hours_ago': int(age.total_seconds() // 3600)
                    })
            except Exception as e:
                # Se fallisce il parsing della data, saltiamo la news
                continue

        if not valid_news:
            print(f"   > Nessuna notizia trovata nelle ultime 48 ore.")
            return 0.0
        else:
            print(f"   > Trovate {len(valid_news)} notizie fresche (ultime 48h):")
            # Stampiamo solo le prime 3 per pulizia
            for n in valid_news[:3]:
                print(f"     📰 [{n['hours_ago']}h fa] {n['title']}")
            
            if len(valid_news) > 3:
                print(f"     ...e altre {len(valid_news)-3}.")

            # --- SIMULAZIONE SENTIMENT ---
            # Qui andrà la tua AI. Ora simulo un valore coerente.
            sim_sent = round(random.uniform(-0.9, 0.9), 2)
            return sim_sent

    except Exception as e:
        print(f"   [Errore RSS] {ticker}: {e}")
        return 0.0

# --- 3. ESECUZIONE ---

if __name__ == "__main__":
    
    env_tickers = os.environ.get('TICKER_INPUT')
    if env_tickers:
        ASSETS = [t.strip() for t in env_tickers.split(',') if t.strip()]
    else:
        ASSETS = ['AAPL', 'NVDA', 'UCG.MI', 'TSLA', 'BTC-USD', 'AMZN']
    
    scorer = HybridScorer()
    
    print(f"\n--- TEST PREVISIONALE RSS ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---")
    

    for ticker in ASSETS:
        print(f"📊 Analisi Asset: {ticker}")
        print("-" * 40)
        
        # 1. Prezzi (yfinance)
        df = get_historical_data(ticker)
        
        if df.empty:
            print("   [!] Dati storici insufficienti.")
            continue
            
        # 2. News (Google RSS)
        # Aggiungiamo un piccolo delay per non far arrabbiare Google
        time.sleep(1) 
        sentiment_score = get_rss_news_sentiment(ticker)
        
        # 3. Calcolo
        result = scorer.calculate_probability(df, sentiment_score)
        
        # 4. Output
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
        print("=" * 40 + "\n")
