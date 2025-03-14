import requests
from bs4 import BeautifulSoup

# URL della pagina di CityFalcon per Tesla
url = 'https://www.cityfalcon.ai/news/directory/stocks/tesla-tsla/news'

# Fare la richiesta HTTP per ottenere la pagina
response = requests.get(url)

# Verifica che la richiesta sia stata eseguita con successo
if response.status_code == 200:
    # Usa BeautifulSoup per analizzare il contenuto HTML
    soup = BeautifulSoup(response.content, 'html.parser')

    # Cerca l'elemento che contiene il sentiment, usando la classe fornita
    sentiment_element = soup.find('span', class_='styles-module__range_value___O89bM')  # Usa la classe che hai trovato

    # Verifica se l'elemento è stato trovato
    if sentiment_element:
        sentiment_text = sentiment_element.get_text()  # Ottieni il testo, che sarà tipo "11%"
        print("Sentiment trovato:", sentiment_text)
    else:
        print("Sentiment non trovato")
else:
    print(f"Errore nel caricamento della pagina: {response.status_code}")
