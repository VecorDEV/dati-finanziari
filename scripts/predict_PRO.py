import requests

# URL della pagina di CityFalcon per Tesla
url = 'https://www.cityfalcon.ai/news/directory/stocks/tesla-tsla/news'

# Fare la richiesta HTTP
response = requests.get(url)

# Verifica che la richiesta sia stata eseguita con successo
if response.status_code == 200:
    # Stampa i primi 1000 caratteri del contenuto HTML per analizzare la risposta
    print(response.text[:1000])
else:
    print(f"Errore nel caricamento della pagina: {response.status_code}")
