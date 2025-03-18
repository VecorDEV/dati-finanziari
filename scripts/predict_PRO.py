import spacy
from github import Github, GithubException
import re
import feedparser
import os

# Carichiamo il modello di lingua inglese
nlp = spacy.load("en_core_web_sm")


def get_stock_news(symbol):
    """ Recupera i titoli delle notizie per un determinato simbolo. """
    url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    
    titles = [entry.title for entry in feed.entries]
    return titles


# Dizionario personalizzato di parole con i punteggi di sentiment
sentiment_dict = {
    # Parole Positive
    "ai": 0.5,  
    "boom": 0.8,  
    "booming": 1.0,  
    "earnings": 0.9,  
    "expansion": 0.6,  
    "highs": 0.8,  
    "innovation": 0.7,  
    "outperformance": 1.0,  
    "outlook": 0.7,  
    "profits": 0.9,  
    "record": 1.0,  
    "revenue": 0.9,  
    "robust": 0.8,  

    # Parole Negative
    "challenges": -0.6,  
    "crisis": -1.0,  
    "decline": -0.7,  
    "delays": -0.7,  
    "faces": 0.2,  
    "fall": -0.7,  
    "issues": -0.6,  
    "lagging": -0.6,  
    "liabilities": -0.7,  
    "losses": -0.8,  
    "market": -0.6,  
    "slowdown": -0.7,  
    "solid": 0.6,  
    "struggles": -0.7,  
    "underperform": -0.8,  
    "weakness": -0.7,  
    "volatility": -0.8,  
    "weak": -0.8,  
    "uncertainty": -0.8,  
    "insolvency": -1.0,  
    "debt": -0.9,  
    "bankruptcy": -1.0,  
    "disappointment": -0.8,  
    "lag": -0.6,  
    "saturation": -0.6,  

    # Avverbi (modificatori)
    "extremely": 1.5,  # Intensificatore forte
    "strongly": 1.5,  # Intensificatore forte
    "highly": 1.5,  # Intensificatore forte
    "severely": 1.5,  # Intensificatore forte
    "significantly": 1.5,  # Intensificatore
    "moderately": 0.5,  # Attenuatore
    "steadily": 1.2,  # Intensificatore moderato
    "somewhat": 0.5,  # Attenuatore
    "slightly": 0.5,  # Attenuatore
    "unexpectedly": 1.0,  # Neutrale (non altera, ma può suggerire un cambiamento inaspettato)
    "eventually": 1.0,  # Neutrale
    "increasingly": 1.2,  # Intensificatore
    "gradually": 1.0,  # Intensificatore (ma più morbido)
    "dramatically": 1.5,  # Intensificatore forte
    "rapidly": 1.5,  # Intensificatore forte
    "sharply": 1.5,  # Intensificatore forte
    "substantially": 1.5,  # Intensificatore forte
    "slightly": 0.5,  # Attenuatore
}

def calculate_sentiment(titles):
    total_sentiment = 0
    total_absolute_sentiment = 0  # Somma assoluta dei punteggi (per la normalizzazione)

    for title in titles:
        print(f"Analyzing: {title}\n")
        
        # Elaboriamo il testo con spaCy
        doc = nlp(title)

        title_sentiment = 0  # Sentiment totale della notizia

        # Analizziamo le parole nel testo
        for token in doc:
            # Verifica se la parola è nel dizionario di sentiment
            if token.text.lower() in sentiment_dict:
                word_sentiment = sentiment_dict[token.text.lower()]
                total_absolute_sentiment += abs(word_sentiment)  # Sommiamo l'assoluto del punteggio
                
                # Stampa la parola e il suo punteggio
                print(f"Word: {token.text} - Sentiment: {word_sentiment}")

                # Se è un nome modificato da un aggettivo, modifichiamo il punteggio
                if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                    for child in token.children:
                        if child.pos_ == "ADJ" or child.pos_ == "ADV":
                            adj_or_adv_sentiment = sentiment_dict.get(child.text.lower(), 0)
                            print(f"  - {child.text} modifies {token.text} with sentiment: {adj_or_adv_sentiment}")
                            word_sentiment += adj_or_adv_sentiment
                
                # Gestiamo gli avverbi che intensificano il verbo (come "sharply")
                if token.pos_ == "VERB" and any(child.pos_ == "ADV" for child in token.children):
                    for child in token.children:
                        if child.pos_ == "ADV" and child.text.lower() in sentiment_dict:
                            word_sentiment *= sentiment_dict[child.text.lower()]
                            print(f"  - {child.text} intensifies {token.text} with multiplier")
                
                # Sommiamo il punteggio del token alla notizia
                title_sentiment += word_sentiment

        # Normalizziamo il punteggio per farlo rientrare tra -1 e 1
        if total_absolute_sentiment > 0:
            normalized_sentiment = title_sentiment / total_absolute_sentiment
            # Limitiamo il punteggio tra -1 e 1
            normalized_sentiment = max(-1, min(1, normalized_sentiment))
        else:
            normalized_sentiment = 0  # Nel caso in cui non ci siano parole con punteggio
        
        # Stampiamo il sentiment totale della notizia
        print(f"Overall Sentiment for this title: {normalized_sentiment}\n")
        
        # Restituiamo la notizia e il sentiment finale
        print(f"Final Sentiment: {title} - {normalized_sentiment}\n")
    

# Recuperiamo i titoli delle notizie per "TSLA" (come esempio)
titles = get_stock_news("TSLA")

# Calcoliamo il sentiment per ogni titolo
calculate_sentiment(titles)
