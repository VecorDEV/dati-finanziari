import os
from datetime import datetime
from google import genai
from google.genai import types

# Inizializza il client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Genera un ID univoco basato su data e ora
notification_id = f"market_alert_{datetime.now().strftime('%d%m%Y_%H%M')}"

# Prompt evoluto per massima qualità e localizzazione
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

try:
    # Usiamo gemini-2.0-flash per il perfetto equilibrio tra intelligenza e precisione nell'output
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],
            temperature=0.3, # Bassa temperatura per maggiore coerenza e precisione tecnica
        )
    )
    
    html_content = response.text.strip()
    
    # Pulizia avanzata per garantire HTML puro
    for cleaner in ["```html", "```"]:
        html_content = html_content.replace(cleaner, "")
    html_content = html_content.strip()

    # Gestione file e cartelle
    folder_path = "interact"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, "push_notifications.html")
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✅ Notifica '{notification_id}' generata con successo in 17 lingue.")

except Exception as e:
    print(f"❌ Errore critico: {e}")
    exit(1)
