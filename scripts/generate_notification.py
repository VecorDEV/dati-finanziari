import os
import time
import re
import random
from datetime import datetime
from google import genai
from google.genai import types

# 1. Configurazione Client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Generazione ID UNIVOCO (Nome + Timestamp al secondo)
unique_id = f"market_alert_{datetime.now().strftime('%d%m%Y_%H%M%S')}"

# 2. DEFINIZIONE DEI FOCUS DINAMICI
# Questo array garantisce che le notifiche varino costantemente argomento
focus_categories = [
    "Titoli Tech e Intelligenza Artificiale (es. NVIDIA, Apple, Microsoft, TSMC, earnings report tech)",
    "Macroeconomia e Banche Centrali (Decisioni FED/BCE, dati su inflazione, tassi d'interesse, disoccupazione USA)",
    "Criptovalute e Blockchain (Bitcoin, Ethereum, approvazione ETF, regolamentazioni, movimenti anomali)",
    "Materie prime ed Energia (Petrolio, Oro, Gas naturale, o tensioni geopolitiche che influenzano direttamente questi prezzi)",
    "Azionario Globale e Finanza (Trimestrali sorprendenti di aziende non-tech, fusioni, acquisizioni, crolli bancari o di specifici settori)"
]

# Python sceglie un argomento a caso per questa specifica esecuzione
current_focus = random.choice(focus_categories)

prompt = f"""
Agisci come un Senior Financial Editor e Localizzazione Expert.
OBIETTIVO:
1. Trova una notizia finanziaria o di mercato molto importante delle ultime 6 ore (Market Movers).
2. FONDAMENTALE: Per questa notifica, devi concentrare la tua ricerca ESCLUSIVAMENTE su questo settore: {current_focus}.
3. Crea una notifica push "Breaking News" ad alto impatto. 
4. Traduci e ADATTA la notifica per 17 mercati internazionali.

REGOLE DI SCRITTURA:
- Tono: Professionale, urgente, autorevole (stile Bloomberg/Financial Times).
- Contenuto: Fatti concreti. Se la notizia riguarda un'azienda specifica (es. NVIDIA), inserisci SEMPRE il nome dell'azienda e le cifre o percentuali rilevanti.
- Lunghezza: Titolo max 50 caratteri, Descrizione max 140 caratteri.
- IMPORTANTE: Scrivi il TITOLO SEMPRE IN TUTTO MAIUSCOLO (ALL CAPS).
- NON inserire emoji nei titoli o nelle descrizioni.

REGOLE DI TRADUZIONE (LOCALIZATION):
- Non tradurre letteralmente. Usa il gergo finanziario corretto per ogni lingua.
- Distingui chiaramente tra Portoghese Europeo (pt) e Brasiliano (pt-BR).

LINGUE RICHIESTE:
Italiano (it), Inglese (en), Spagnolo (es), Francese (fr), Tedesco (de), Portoghese EU (pt), Portoghese Brasiliano (pt-BR), Olandese (nl), Russo (ru), Ucraino (uk), Cinese Semplificato (zh-CN), Giapponese (ja), Coreano (ko), Indonesiano (id), Hindi (hi), Arabo (ar), Polacco (pl).

OUTPUT RICHIESTO:
Restituisci ESCLUSIVAMENTE il codice HTML. Non aggiungere commenti o markdown.
Assicurati che l'attributo data-type sia sempre impostato su "news".

TEMPLATE DA SEGUIRE SCRUPOLOSAMENTE:
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>Repository Notifiche App</title>
</head>
<body>
    <div class="notification" id="SEGNAPOSTO" lang="ISO_CODE" data-target="all" data-type="news" data-link="">
        <h3>[TITOLO IN TUTTO MAIUSCOLO]</h3>
        <p>[DESCRIZIONE]</p>
    </div>
</body>
</html>
"""

def generate_with_retry(max_retries=5, initial_delay=10):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            print(f"🔄 Tentativo {attempt + 1}/{max_retries} (Focus: {current_focus})...")
            
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.5, # Aumentato a 0.5 per consentire maggiore varietà nella scelta delle notizie
                )
            )
            
            if not response or not response.text:
                raise ValueError("Risposta vuota dal modello")
                
            return response.text.strip()

        except Exception as e:
            err_msg = str(e).upper()
            print(f"⚠️ Errore: {e}")
            if any(x in err_msg for x in ["503", "429", "404", "UNAVAILABLE", "EXHAUSTED", "NONE"]):
                if attempt < max_retries - 1:
                    print(f"🕒 Riprovo in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
            raise e

try:
    # 2. Generazione
    html_raw = generate_with_retry()
    
    # 3. Pulizia Markdown (A PROVA DI COPIA-INCOLLA)
    clean_content = html_raw
    md_start = "`" * 3 + "html"
    md_end = "`" * 3
    
    if md_start in clean_content:
        clean_content = clean_content.split(md_start)[-1].split(md_end)[0]
    elif md_end in clean_content:
        clean_content = clean_content.split(md_end)[-1].split(md_end)[0]
    
    html_content = clean_content.strip()
    
    # 4. SOSTITUZIONE AGGRESSIVA DEGLI ID TRAMITE REGEX
    html_content = re.sub(r'id="[^"]+"', f'id="{unique_id}"', html_content)

    # 5. Salvataggio
    folder_path = "interact"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, "push_notifications.html")
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✅ File aggiornato con successo. Argomento trattato: {current_focus}")

except Exception as e:
    print(f"❌ Errore critico finale: {e}")
    exit(1)
