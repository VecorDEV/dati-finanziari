import os
import google.generativeai as genai
from datetime import datetime

# Configurazione API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel(
    model_name='gemini-1.5-pro',
    tools='google_search_retrieval'
)

notification_id = f"market_alert_{datetime.now().strftime('%d%m%Y_%H%M')}"

prompt = f"""
Cerca la notizia finanziaria/geopolitica più importante di OGGI. 
Genera il file HTML con le traduzioni richieste (it, en, es, fr, de, pt, pt-BR, nl, ru, uk, zh-CN, ja, ko, id, hi, ar, pl).
Usa l'ID: {notification_id}.
Restituisci solo il codice HTML, senza blocchi di testo o markdown.
"""

try:
    response = model.generate_content(prompt)
    html_content = response.text.strip()
    
    if html_content.startswith("```html"):
        html_content = html_content[7:]
    if html_content.endswith("
```"):
        html_content = html_content[:-3]

    # --- NUOVA LOGICA PER LA CARTELLA INTERACT ---
    folder_path = "interact"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Cartella '{folder_path}' creata.")

    file_path = os.path.join(folder_path, "notifica_odierna.html")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✅ File salvato in: {file_path}")

except Exception as e:
    print(f"❌ Errore: {e}")
    exit(1)
