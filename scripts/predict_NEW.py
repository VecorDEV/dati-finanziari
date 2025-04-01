from textblob import TextBlob

# Funzione per valutare il sentiment
def evaluate_sentiment_textblob(text):
    # Crea un oggetto TextBlob
    blob = TextBlob(text)
    
    # Ottieni il polare sentiment (range: -1 negativo, 1 positivo)
    sentiment_score = blob.sentiment.polarity
    
    return sentiment_score

# Test del codice con la frase fornita
text = """Analysts anticipate that Tesla’s first-quarter vehicle deliveries will be the lowest in over two years, with estimates around 377,592 units. This projection is attributed to several factors, including production challenges with the Model Y, public backlash against CEO Elon Musk’s political involvement, and consumer delays in anticipation of a lower-priced model. Stifel analyst Stephen Gengaro reduced his delivery forecast by 23%, highlighting concerns over Musk’s association with President Donald Trump and its impact on sales."""

sentiment_score = evaluate_sentiment_textblob(text)

# Stampa il punteggio di sentiment
print("Sentiment Score:", sentiment_score)

# Interpretazione
if sentiment_score > 0:
    print("Sentiment: Positive")
elif sentiment_score < 0:
    print("Sentiment: Negative")
else:
    print("Sentiment: Neutral")
