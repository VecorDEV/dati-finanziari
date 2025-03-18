import spacy

# Carichiamo il modello di lingua inglese
nlp = spacy.load("en_core_web_sm")

# Dizionario personalizzato di parole con i punteggi di sentiment
sentiment_dict = {
    "demand": 0.7,  # Positivo
    "slower": -0.5,  # Negativo
    "ai": 0.5,  # Positivo
    "expansion": 0.6,  # Positivo
    "growth": 0.8,  # Positivo
    "production": 0.4,  # Positivo
    "delays": -0.7,  # Negativo
    "stock": 0.4,  # Positivo
    "earnings": 0.9,  # Positivo
    "plunges": -0.8,  # Molto negativo
    "highs": 0.8,  # Positivo
    "record": 0.9,  # Positivo
    "issues": -0.6,  # Negativo
    "hits": 0.5,  # Positivo
    "drops": -0.8,  # Negativo
    "sharply": 1.5,  # Intensificatore
    "solid": 0.6,  # Positivo
    "faces": 0.2,  # Neutrale, ma associato a sfide, quindi negativo
    "challenges": -0.6,  # Negativo
    "supply": -0.7,  # Negativo
    "chain": -0.7,  # Negativo
    "market": -0.6,  # Negativo
    "saturation": -0.6,  # Negativo
    "reports": 0.3,  # Positivo
    "new": 0.0,  # Neutrale
    "data": 0.4,  # Positivo
    "center": 0.4,  # Positivo
}

# Funzione per calcolare il punteggio del sentiment
def calculate_sentiment(doc):
    total_sentiment = 0
    total_absolute_sentiment = 0  # Somma assoluta dei punteggi (per la normalizzazione)
    
    # Analizziamo le parole nel testo
    for token in doc:
        # Verifica se la parola è nel dizionario di sentiment
        if token.text.lower() in sentiment_dict:
            word_sentiment = sentiment_dict[token.text.lower()]
            total_absolute_sentiment += abs(word_sentiment)  # Sommiamo l'assoluto del punteggio
            
            # Se è un nome modificato da un aggettivo, modifichiamo il punteggio
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                # Controlliamo se l'aggettivo (aggettivi, avverbi) lo modifica
                for child in token.children:
                    if child.pos_ == "ADJ" or child.pos_ == "ADV":
                        # Modifichiamo il punteggio in base alla relazione
                        word_sentiment += sentiment_dict.get(child.text.lower(), 0)
            
            # Gestiamo gli avverbi che intensificano il verbo (come "sharply")
            if token.pos_ == "VERB" and any(child.pos_ == "ADV" for child in token.children):
                for child in token.children:
                    if child.pos_ == "ADV" and child.text.lower() in sentiment_dict:
                        word_sentiment *= sentiment_dict[child.text.lower()]
            
            # Sommiamo il punteggio del token
            total_sentiment += word_sentiment

    # Normalizziamo il punteggio per farlo rientrare tra -1 e 1
    if total_absolute_sentiment > 0:
        normalized_sentiment = total_sentiment / total_absolute_sentiment
        # Limitiamo il punteggio tra -1 e 1
        normalized_sentiment = max(-1, min(1, normalized_sentiment))
    else:
        normalized_sentiment = 0  # Nel caso in cui non ci siano parole con punteggio

    return normalized_sentiment

# Testo di esempio
text = "Nvidia Hits New Highs as AI Expansion and Data Center Growth Drive Record Earnings"

# Elaboriamo il testo con spaCy
doc = nlp(text)

# Calcoliamo il sentiment
sentiment_score = calculate_sentiment(doc)

print("Sentiment score:", sentiment_score)
