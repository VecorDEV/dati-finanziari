from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

# Configura le opzioni per il browser headless
chrome_options = Options()
chrome_options.add_argument('--headless')  # Esegui in modalità headless (senza interfaccia grafica)

# Configura il driver di Selenium
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# URL della pagina di CityFalcon per Tesla
url = 'https://www.cityfalcon.ai/news/directory/stocks/tesla-tsla/news'

# Apri la pagina
driver.get(url)

# Aspetta che la pagina venga caricata completamente (modifica il tempo se necessario)
time.sleep(5)

# Trova l'elemento che contiene la percentuale del sentiment usando la classe identificata
sentiment_element = driver.find_element(By.CLASS_NAME, 'styles-module__range_value___O89bM')

# Ottieni il testo del sentiment
sentiment_text = sentiment_element.text

# Stampa il risultato
print("Sentiment trovato:", sentiment_text)

# Chiudi il driver
driver.quit()
