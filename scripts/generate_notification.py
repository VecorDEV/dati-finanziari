import os
from datetime import datetime
from google import genai
from google.genai import types

# Inizializza il nuovo client usando la chiave API
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

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
Sostituisci solo il TITOLO, il TESTO DELLA NOTIFICA e l'attributo lang. Usa l'ID: {notification_id}. 
IMPORTANTE: Restituisci SOLO codice puro, senza alcuna formattazione, blocchi di codice o tag markdown.

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
    # Usiamo la nuova sintassi per generare il contenuto con grounding su Google Search
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],
        )
    )
    
    html_content = response.text.strip()
    
    # Pulizia sicura del markdown
    html_content = html_content.replace('`' * 3 + 'html\n', '')
    html_content = html_content.replace('`' * 3 + 'html', '')
    html_content = html_content.replace('`' * 3, '')
    html_content = html_content.strip()
        
    # Creazione cartella e salvataggio
    folder_path = "interact"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Cartella '{folder_path}' creata.")

    file_path = os.path.join(folder_path, "notifica_odierna.html")
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✅ File HTML generato con successo e salvato in: {file_path}")

except Exception as e:
    print(f"❌ Errore durante la generazione: {e}")
    exit(1)
