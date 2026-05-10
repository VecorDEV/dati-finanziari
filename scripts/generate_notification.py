import os
import time
from datetime import datetime
from google import genai
from google.genai import types

# Inizializza il client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Genera un ID univoco basato su data e ora
notification_id = f"market_alert_{datetime.now().strftime('%d%m%Y_%H%M')}"

# Prompt evoluto per massima qualità, localizzazione e sicurezza copyright
prompt = f"""
Agisci come un Senior Financial Editor e Localizzazione Expert.
OBIETTIVO:
1. Trova la notizia finanziaria o geopolitica più impattante delle ultime 6 ore (Market Movers).
2. Crea una notifica push "Breaking News" ad alto impatto. 
3. Traduci e ADATTA la notifica per 17 mercati internazionali.

REGOLE DI SCRITTURA:
- Tono: Professionale, urgente, autorevole (stile Bloomberg/Financial Times).
- Contenuto: Fatti concreti, cifre se disponibili, impatto chiaro sui mercati.
- Lunghezza: Titolo max 50 caratteri, Descrizione max 140 caratteri.
- ORIGINALE E FAIR USE: Mai copiare intere frasi dalle fonti trovate; rielabora sempre i fatti con parole tue in modo originale e sintetico.

REGOLE DI TRADUZIONE (LOCALIZATION):
- Non tradurre letteralmente. Usa il gergo finanziario corretto per ogni lingua (es. in italiano usa 'azionario' o 'listini', non 'mercato dei titoli').
- Distingui chiaramente tra Portoghese Europeo (pt) e Brasiliano (pt-BR).
- Per il mercato Arabo e Cinese, usa un tono formale e rispettoso delle convenzioni locali.

ID UNIVOCO DA USARE: notification_id

LINGUE RICHIESTE:
Italiano (it), Inglese (en), Spagnolo (es), Francese (fr), Tedesco (de), Portoghese EU (pt), Portoghese Brasiliano (pt-BR), Olandese (nl), Russo (ru), Ucraino (uk), Cinese Semplificato (zh-CN), Giapponese (ja), Coreano (ko), Indonesiano (id), Hindi (hi), Arabo (ar), Polacco (pl).

OUTPUT RICHIESTO:
Restituisci ESCLUSIVAMENTE il codice HTML. Non aggiungere commenti, non aggiungere ```html.
Ogni lingua deve avere il suo div con l'attributo lang corrispondente.

TEMPLATE DA SEGUIRE SCRUPOLOSAMENTE:
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>Repository Notifiche App</title>
</head>
<body>
    <!-- Genera un div per ogni lingua richiesta -->
    <div class="notification" id="notification_id" lang="[ISO_CODE]" data-target="all" data-link="">
        <h3>[TITOLO]</h3>
        <p>[DESCRIZIONE]</p>
    </div>
</body>
</html>
"""

# Funzione a prova di bomba per gestire i sovraccarichi dei server (503 / 429)
def generate_with_retry(max_retries=3, base_delay=10):
    delay = base_delay
    for attempt in range(max_retries):
        try:
            print(f"🔄 Tentativo {attempt + 1} di {max_retries}...")
            
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.3, # Bassa temperatura per maggiore coerenza e precisione tecnica
                )
            )
            return response.text.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "429" in error_msg or "Unavailable" in error_msg:
                print(f"⚠️ Server occupato o limiti di traffico. Riprovo tra {delay} secondi...")
                time.sleep(delay)
                delay *= 2 # Exponential backoff: aspetta 10s, poi 20s, ecc.
            else:
                # Se è un errore di codice o di chiave API, interrompe subito
                raise e
                
    raise Exception("❌ Tutti i tentativi di generazione sono falliti a causa del sovraccarico del server.")

# Esecuzione principale
try:
    # 1. Chiama la funzione con retry
    html_content = generate_with_retry()
    
    # 2. Pulizia avanzata per garantire HTML puro
    for cleaner in ["
```html", "```"]:
        html_content = html_content.replace(cleaner, "")
    html_content = html_content.strip()

    # 3. Gestione file e cartelle
    folder_path = "interact"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Assicuriamoci che il nome corrisponda a quello che l'app scarica
    file_path = os.path.join(folder_path, "push_notifications.html")
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✅ Notifica '{notification_id}' generata con successo e salvata in {file_path}")

except Exception as e:
    print(f"❌ Errore critico non recuperabile: {e}")
    exit(1)
