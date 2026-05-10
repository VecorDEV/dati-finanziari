import os
import time
import shutil
from datetime import datetime
from google import genai
from google.genai import types

# 1. Configurazione Iniziale
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
notification_id = f"market_alert_{datetime.now().strftime('%d%m%Y_%H%M')}"

# Il tuo prompt (invariato come richiesto)
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

REGOLE DI TRADUZIONE (LOCALIZATION):
- Non tradurre letteralmente. Usa il gergo finanziario corretto per ogni lingua (es. in italiano usa 'azionario' o 'listini', non 'mercato dei titoli').
- Distingui chiaramente tra Portoghese Europeo (pt) e Brasiliano (pt-BR).
- Per il mercato Arabo e Cinese, usa un tono formale e rispettoso delle convenzioni locali.

ID UNIVOCO DA USARE: {notification_id}

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
    <div class="notification" id="notification_id" lang="ISO_CODE" data-target="all" data-link="">
        <h3>[TITOLO]</h3>
        <p>[DESCRIZIONE]</p>
    </div>
</body>
</html>
"""

def generate_with_retry(max_retries=5, initial_delay=10):
    """Esegue la chiamata a Gemini con logica di retry per errori 503 e 429."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            print(f"🔄 Tentativo {attempt + 1}/{max_retries}...")
            
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.3,
                )
            )
            return response.text.strip()

        except Exception as e:
            err_msg = str(e).upper()
            # Se è un errore di disponibilità o quota, riprova
            if "503" in err_msg or "429" in err_msg or "UNAVAILABLE" in err_msg or "EXHAUSTED" in err_msg:
                if attempt < max_retries - 1:
                    print(f"⚠️ Server occupato o limite raggiunto. Riprovo in {delay}s...")
                    time.sleep(delay)
                    delay *= 2 # Raddoppia l'attesa (Exponential Backoff)
                    continue
            raise e

try:
    # 2. Generazione del contenuto
    html_content = generate_with_retry()
    
    # 3. Pulizia stringhe markdown
    for cleaner in ["```html", "```"]:
        html_content = html_content.replace(cleaner, "")
    html_content = html_content.strip()

    # 4. Salvataggio sicuro nel file
    folder_path = "interact"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Cartella '{folder_path}' creata.")

    # Nome file allineato a quello che si aspetta la tua app
    file_path = os.path.join(folder_path, "push_notifications.html")
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✅ Notifica '{notification_id}' pubblicata con successo in {file_path}")

except Exception as e:
    print(f"❌ Errore critico dopo tutti i tentativi: {e}")
    exit(1)
