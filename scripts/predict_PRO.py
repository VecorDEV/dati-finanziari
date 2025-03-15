from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
time.sleep(5)  # Puoi aumentare questo tempo se il caricamento della pagina è lento

try:
    # Aspetta fino a quando l'elemento sentiment è visibile (modifica la classe se necessario)
    sentiment_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'styles-module__range_value___089bM'))
    )
    
    # Ottieni il testo del sentiment
    sentiment_text = sentiment_element.text
    print("Sentiment trovato:", sentiment_text)
    
except Exception as e:
    print("Errore durante la ricerca del sentiment:", e)

# Chiudi il driver
driver.quit()
