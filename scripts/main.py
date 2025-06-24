import pandas as pd
from gdeltdoc import GdeltDoc, Filters
from pysentimiento import SentimentAnalyzer

# Inizializza l'analizzatore del sentiment
analyzer = SentimentAnalyzer()

# Funzione per ottenere articoli relativi a un simbolo
def get_articles(symbol, start_date, end_date):
    filters = Filters(keyword=symbol, start_date=start_date, end_date=end_date)
    gd = GdeltDoc()
    articles = gd.article_search(filters)
    return articles

# Funzione per analizzare il sentiment degli articoli
def analyze_sentiment(articles):
    sentiments = []
    for _, row in articles.iterrows():
        text = row['title'] + ' ' + row['url']
        sentiment = analyzer.predict(text)
        sentiments.append(sentiment)
    articles['sentiment'] = sentiments
    return articles

# Esempio di utilizzo
symbol = 'AAPL'
start_date = '2025-06-01'
end_date = '2025-06-24'
articles = get_articles(symbol, start_date, end_date)
articles_with_sentiment = analyze_sentiment(articles)
print(articles_with_sentiment[['title', 'sentiment']])
