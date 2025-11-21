import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
import time
import random

# --- 1. IL MODELLO IBRIDO (La "Piramide") ---
class HybridScorer:
    def _calculate_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _get_technical_score(self, df):
        if len(df) < 200: return 0.0
        
        close = df['Close']
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        sma_200 = close.rolling(window=200).mean().iloc[-1]
        rsi = self._calculate_rsi(close).iloc[-1]
        current_price = close.iloc[-1]

        score = 0.0
        
        # Logica Tecnica Semplificata
        if current_price > sma_200: score += 0.5
        else: score -= 0.5
        
        if rsi < 30: score += 0.5
        elif rsi > 70: score -= 0.5
        
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df_history, news_sentiment):
        s_tech = self._get_technical_score(df_history)
        s_news = news_sentiment
        
        # LOGICA DI PESATURA DINAMICA
        # Se sentiment è 0.0 (nessuna news OGGI), il peso news diventa 0
        has_news = abs(s_news) > 0.01 

        if has_news:
            w_news = 0.50
            w_tech = 0.50
            mode = "IBRIDA (News + Tech)"
        else:
            w_news = 0.00
            w_tech = 1.00 # Se non ci sono news, il tecnico comanda al 100%
            mode = "SOLO TECNICA (No News Oggi)"
        
        final_score = (s_news * w_news) + (s_tech * w_tech)
        probability = 50 + (final_score * 50)

        return {
            'probability': round(probability, 2),
            'mode': mode,
            'details': {'tech_score': round(s_tech, 2), 'news_score': round(s_news, 2)}
        }

# --- 2. FUNZIONI DI SUPPORTO ---

def get_historical_data(ticker):
    """Recupera 1 anno di dati storici."""
    try:
        stock = yf.Ticker(ticker)
        # Scarichiamo 1 anno per calcolare la SMA200
        df = stock.history(period="1y") 
        return df
    except Exception as e:
        print(f"Errore storico {ticker}: {e}")
        return pd.DataFrame()

def get_todays_news_sentiment(ticker):
    """
    Recupera le news e FILTRA SOLO quelle di OGGI.
    Restituisce un sentiment simulato (tu integrerai il tuo vero NLP qui).
    """
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        
        today_news = []
        today_date = date.today()
        
        print(f"   > Controllo notizie per {ticker}...")
        
        for news in news_list:
            # yfinance restituisce il timestamp in 'providerPublishTime'
            if 'providerPublishTime' in news:
                pub_time = news['providerPublishTime']
                pub_date = datetime.fromtimestamp(pub_time).date()
                
                # === IL FILTRO CRUCIALE ===
                # Teniamo la notizia SOLO se la data è uguale a OGGI
                if pub_date == today_date:
                    today_news.append(news['title'])
                else:
                    # Debug opzionale per vedere cosa scartiamo
                    # print(f"     - Scartata notizia vecchia del {pub_date}")
                    pass
        
        # Calcolo Sentiment sulle notizie filtrate
        if not today_news:
            print(f"   > Nessuna notizia trovata per OGGI ({today_date}).")
            return 0.0 # Neutro
        else:
            print(f"   > Trovate {len(today_news)} notizie fresche di giornata!")
            for t in today_news:
                print(f"     * {t}")
            
            # QUI INSERISCI IL TUO ALGORITMO DI SENTIMENT REALE
            # Per questo test, simulo un valore casuale tra -1 e 1
            # così vedi come reagisce il modello.
            simulated_sentiment = round(random.uniform(-0.8, 0.8), 2)
            return simulated_sentiment

    except Exception as e:
        print(f"Errore news {ticker}: {e}")
        return 0.0

# --- 3. ESECUZIONE PRINCIPALE ---

if __name__ == "__main__":
    
    # Lista degli asset da testare
    ASSETS = ['AAPL', 'NVDA', 'UCG.MI', 'TSLA']
    
    scorer = HybridScorer()
    
    print(f"--- AVVIO TEST PREVISIONALE ({date.today()}) ---")
    

    for ticker in ASSETS:
        print(f"\nAnalisi Asset: {ticker}")
        print("-" * 30)
        
        # 1. Dati Storici
        df = get_historical_data(ticker)
        if df.empty:
            print("Dati storici non disponibili.")
            continue
            
        # 2. News Sentiment (Filtrato per Oggi)
        sentiment_score = get_todays_news_sentiment(ticker)
        
        # 3. Calcolo Probabilità (Modello Ibrido)
        result = scorer.calculate_probability(df, sentiment_score)
        
        # 4. Log Risultati
        prob = result['probability']
        signal = "COMPRA 🟢" if prob > 55 else "VENDI 🔴" if prob < 45 else "NEUTRO ⚪"
        
        print(f"\n   RISULTATO FINALE:")
        print(f"   Probabilità Rialzo: {prob}%")
        print(f"   Segnale: {signal}")
        print(f"   Modalità Usata: {result['mode']}")
        print(f"   Dettagli Punteggi: Tech {result['details']['tech_score']} | News {result['details']['news_score']}")
        print("=" * 30)

