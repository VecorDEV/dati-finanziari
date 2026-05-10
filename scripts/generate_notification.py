import os
import google.generativeai as genai
from datetime import datetime

# Configura le API di Gemini usando la chiave salvata nei secrets di GitHub
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Utilizziamo Gemini 1.5 Pro abilitando lo strumento di ricerca Google
# per garantire che le notizie siano "freschissime" e di giornata.
model = genai.GenerativeModel(
    model_name='gemini-1.5-pro',
    tools='google_search_retrieval'
)

today_str = datetime.now().strftime("%B %Y").lower()
notification_id = f"market_alert_{datetime.now().strftime('%d%m%Y_%H%M')}"

prompt = f"""
Sei un analista finanziario esperto e un copywriter per un'app internazionale.
Cerca sul web la notizia finanziaria, macroeconomica o geopolitica più importante e impattante di OGGI. Deve essere un evento reale, accaduto nelle ultime ore (es. decisioni FED, dichiarazioni geopolitiche rilevanti, crolli/rally di settori o grandi aziende).

Crea una notifica breve e accattivante (titolo + 1-2 frasi di descrizione) che spinga l'utente ad aprire l'app per controllare i settori influenzati.

Traduci la notifica in queste lingue:
Italiano (it), Inglese (en), Spagnolo (es), Francese (fr), Tedesco (de), Portoghese EU (pt), Portoghese Brasiliano (pt-BR), Olandese (nl), Russo (ru), Ucraino (uk), Cinese Semplificato (zh-CN), Giapponese (ja), Coreano (ko), Indonesiano (id), Hindi (hi), Arabo (ar), Polacco (pl).

Restituisci ESCLUSIVAMENTE codice HTML valido, usando ESATTAMENTE questo template. 
Devi generare un blocco <div class="notification"> PER OGNI LINGUA richiesta, tutti all'interno dello stesso tag <body>.
Sostituisci solo il TITOLO, il TESTO DELLA NOTIFICA e l'attributo lang. Usa l'ID: {notification_id}. Non aggiungere markdown (come ```html), solo il codice puro.

Template:
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>Repository Notifiche App</title>
</head>
<body>

    <!-- Ripeti questo div per ogni lingua richiesta -->
    <div class="notification" 
         id="notification_id" 
         lang="CODICE_LINGUA" 
         data-target="all" 
         data-link="">
         
        <h3>[TITOLO NELLA LINGUA SPECIFICA]</h3>
        <p>[TESTO NELLA LINGUA SPECIFICA]</p>
    </div>

</body>
</html>
"""

try:
    response = model.generate_content(prompt)
    html_content = response.text.strip()
    
    # Rimuove eventuali blocchi markdown se Gemini li inserisce per errore
    if html_content.startswith("
```html"):
        html_content = html_content[7:]
    elif html_content.startswith("```"):
        html_content = html_content[3:]
        
    if html_content.endswith("
```"):
        html_content = html_content[:-3]
        
    html_content = html_content.strip()
        
    # Verifica che la cartella interact esista, altrimenti la crea
    folder_path = "interact"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Cartella '{folder_path}' creata.")

    # Percorso completo del file all'interno della cartella
    file_path = os.path.join(folder_path, "notifica_odierna.html")
        
    # Salva o sovrascrive il risultato nel file HTML
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✅ File HTML generato con successo e salvato in: {file_path}")

except Exception as e:
    print(f"❌ Errore durante la generazione: {e}")
    exit(1)
