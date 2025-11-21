import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import os

# --- 1. IL MODELLO MATEMATICO (Core Logic) ---
class HybridScorer:
    """
    Calcola la probabilità di rialzo basandosi su Analisi Tecnica 
    e (se disponibile) Sentiment delle Notizie recenti.
    """
    
    def _calculate_rsi(self, series, period=14):
        """Calcola l'indicatore RSI manualmante (senza librerie esterne TA)."""
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Evita divisione per zero
        loss = loss.replace(0, np.nan) 
        rs = gain / loss
        rs = rs.fillna(0) # Gestione casi limite
        
        return 100 - (100 / (1 + rs))

    def _get_technical_score(self, df):
        """Restituisce punteggio tecnico da -1.0 a +1.0"""
        if len(df) < 200: 
            return 0.0 # Dati insufficienti
        
        close = df['Close']
        
        # Calcolo indicatori sull'ultimo dato disponibile
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        sma_200 = close.rolling(window=200).mean().iloc[-1]
        rsi = self._calculate_rsi(close).iloc[-1]
        current_price = close.iloc[-1]

        score = 0.0
        
        # --- Logica Tecnica ---
        # 1. Trend Following (Siamo in Bull Market?)
        if current_price > sma_200: 
            score += 0.5
        else: 
            score -= 0.5
        
        # 2. Momentum/RSI (Siamo ipervenduti/ipercomprati?)
        if rsi < 30: 
            score += 0.5 # Probabile rimbalzo
        elif rsi > 70: 
            score -= 0.5 # Probabile ritracciamento
        
        # Normalizza (clamp) tra -1 e 1
        return max(min(score, 1.0), -1.0)

    def calculate_probability(self, df_history, news_sentiment):
        """Combina i punteggi con pesi dinamici."""
        
        # Calcola score tecnico
        s_tech = self._get_technical_score(df_history)
        s_news = news_sentiment
        
        # --- PESATURA DINAMICA ---
        # Controlliamo se il sentiment è attivo (cioè diverso da 0.0)
        # Usiamo una soglia minima (0.01) per evitare float imperfetti
        has_news = abs(s_news) > 0.01 

        if has_news:
            # SCENARIO A: Abbiamo notizie fresche
            # Le news contano molto (50%), il tecnico conferma (50%)
            w_news = 0.50
            w_tech = 0.50
            mode = "IBRIDA (News + Tech)"
        else:
            # SCENARIO B: Silenzio Stampa
            # Ci affidiamo totalmente alla statistica tecnica
            w_news = 0.00
            w_tech = 1.00 
            mode = "SOLO TECNICA (No News < 24h)"
        
        # Somma ponderata
        final_score = (s_news * w_news) + (s_tech * w_tech)
        
        # Conversione in probabilità (range 0-100%)
        probability = 50 + (final_score * 50)

        return {
            'probability': round(probability, 2),
            'mode': mode,
            'details': {
                'tech_score': round(s_tech, 2), 
                'news_score': round(s_news, 2),
                'weights': {'news': w_news, 'tech': w_tech}
            }
        }

# --- 2. FUNZIONI DI RECUPERO DATI ---

def get_historical_data(ticker):
    """Recupera dati storici (1 anno) per calcolo SMA200."""
    try:
        # CORREZIONE: Rimossa 'progress=False' che causava l'errore
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y") 
        return df
    except Exception as e:
        print(f"   [Errore Dati Storici] {ticker}: {e}")
        return pd.DataFrame()

def get_recent_news_sentiment(ticker):
    """
    Recupera news e applica filtro 'Sliding Window' (ultime 24h).
    Restituisce un sentiment simulato (da sostituire con tua AI).
    """
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        
        valid_news = []
        now = datetime.now()
        
        # FINESTRA TEMPORALE: 24 Ore
        # Puoi cambiarlo a 48 o 72 se vuoi includere il weekend
        time_window = timedelta(hours=24)
        
        print(f"   > Ricerca notizie ultime 24h per {ticker}...")
        
        for news in news_list:
            # Verifica se c'è il timestamp di pubblicazione
            if 'providerPublishTime' in news:
                pub_timestamp = news['providerPublishTime']
                pub_date = datetime.fromtimestamp(pub_timestamp)
                
                # Calcola età della notizia
                age = now - pub_date
                
                # --- IL FILTRO ---
                if age < time_window:
                    hours_ago = int(age.total_seconds() // 3600)
                    valid_news.append({
                        'title': news['title'],
                        'age_str': f"{hours_ago}h fa"
                    })
        
        # Logica di Ritorno
        if not valid_news:
            print(f"   > Nessuna notizia trovata nelle ultime 24 ore.")
            return 0.0 # Sentiment Neutro (attiva modalità SOLO TECNICA)
        else:
            print(f"   > Trovate {len(valid_news)} notizie recenti:")
            for n in valid_news:
                print(f"     * [{n['age_str']}] {n['title']}")
            
            # --- SIMULAZIONE SENTIMENT ---
            # QUI INTEGRESTI IL TUO MODELLO NLP REALE
            # Ora generiamo un valore random coerente per testare il modello ibrido
            # (Es. tra -0.8 e 0.8)
            simulated_sentiment = round(random.uniform(-0.9, 0.9), 2)
            print(f"   > (Simulazione) Sentiment Calcolato: {simulated_sentiment}")
            return simulated_sentiment

    except Exception as e:
        print(f"   [Errore News] {ticker}: {e}")
        return 0.0

# --- 3. ESECUZIONE ---

if __name__ == "__main__":
    
    # Gestione Input per GitHub Actions o uso locale
    env_tickers = os.environ.get('TICKER_INPUT')
    if env_tickers:
        # Se passato da GitHub, pulisci la stringa
        ASSETS = [t.strip() for t in env_tickers.split(',') if t.strip()]
    else:
        # Lista di default per test locale
        ASSETS = ['AAPL', 'NVDA', 'UCG.MI', 'TSLA', 'BTC-USD', 'AMZN']
    
    scorer = HybridScorer()
    
    print(f"\n--- AVVIO TEST PREVISIONALE IBRIDO ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---")
    print(f"Asset in analisi: {ASSETS}\n")

    for ticker in ASSETS:
        print(f"📊 Analisi Asset: {ticker}")
        print("-" * 40)
        
        # 1. Recupero Storico
        df = get_historical_data(ticker)
        
        if df.empty:
            print("   [!] Dati storici insufficienti. Salto.")
            continue
            
        # 2. Recupero Sentiment (Ultime 24h)
        sentiment_score = get_recent_news_sentiment(ticker)
        
        # 3. Calcolo Modello
        result = scorer.calculate_probability(df, sentiment_score)
        
        # 4. Formattazione Output
        prob = result['probability']
        
        # Definizione Segnale Visivo
        if prob >= 60:
            signal = "🟢 STRONG BUY"
        elif prob >= 55:
            signal = "🟢 BUY"
        elif prob <= 40:
            signal = "🔴 STRONG SELL"
        elif prob <= 45:
            signal = "🔴 SELL"
        else:
            signal = "⚪ HOLD / NEUTRO"
        
        print(f"\n   🎯 RISULTATO FINALE:")
        print(f"   Segnale: {signal}")
        print(f"   Probabilità Rialzo: {prob}%")
        print(f"   Modalità: {result['mode']}")
        print(f"   Dettagli Pesi: {result['details']['weights']}")
        print("=" * 40 + "\n")



