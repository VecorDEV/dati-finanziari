import os
import sys
import time
import re
import random
from datetime import datetime
from google import genai
from google.genai import types

# 1. Client: il tool google_search richiede l'API v1beta
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(api_version="v1beta"),
)

# Modelli in ordine di preferenza (il primo è quello con quota confermata nel tuo progetto).
# Puoi forzarne uno con la variabile d'ambiente GEMINI_MODEL.
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]
if os.environ.get("GEMINI_MODEL"):
    MODEL_CANDIDATES.insert(0, os.environ["GEMINI_MODEL"])

EXPECTED_NOTIFICATIONS = 17


def log(msg):
    print(msg, flush=True)


# ID UNIVOCO (Nome + Timestamp al secondo)
unique_id = f"market_alert_{datetime.now().strftime('%d%m%Y_%H%M%S')}"

# 2. FOCUS DINAMICI
focus_categories = [
    "Titoli Tech e Intelligenza Artificiale (es. NVIDIA, Apple, Microsoft, TSM, Tesla, earnings report tech)",
    "Macroeconomia e Banche Centrali (Decisioni FED/BCE, dati su inflazione, tassi d'interesse, disoccupazione USA)",
    "Criptovalute e Blockchain (Bitcoin, Ethereum, approvazione ETF, regolamentazioni, movimenti anomali)",
    "Materie prime ed Energia (Petrolio, Oro, Gas naturale, o tensioni geopolitiche che influenzano direttamente questi prezzi)",
    "Azionario Globale e Finanza (Trimestrali sorprendenti di aziende non-tech, fusioni, acquisizioni, crolli bancari o di specifici settori)",
]

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


class SkipModel(Exception):
    """Questo modello non è utilizzabile ora: passa al successivo."""


def call_model(model, max_attempts=2, delay=20):
    for attempt in range(1, max_attempts + 1):
        try:
            log(f"Modello {model} - tentativo {attempt}/{max_attempts} (Focus: {current_focus})")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            text = (response.text or "").strip() if response else ""
            if not text:
                raise ValueError("EMPTY")
            return text
        except Exception as e:
            err = str(e).upper()
            log(f"Errore con {model}: {e}")

            # Modello inesistente/ritirato, argomento non valido o quota esaurita
            # (RPD): inutile insistere, si passa al modello successivo.
            if "404" in err or "NOT_FOUND" in err or "INVALID_ARGUMENT" in err:
                raise SkipModel(str(e))

            # 429: potrebbe essere solo il limite al minuto (RPM), un solo retry
            # dopo una pausa; se persiste si passa al modello successivo.
            retryable = any(x in err for x in ["429", "RESOURCE_EXHAUSTED", "503", "500", "UNAVAILABLE", "DEADLINE", "EMPTY"])
            if retryable and attempt < max_attempts:
                log(f"Riprovo tra {delay}s...")
                time.sleep(delay)
                continue
            raise SkipModel(str(e))


def generate_with_fallback():
    last_error = None
    for model in MODEL_CANDIDATES:
        try:
            return call_model(model)
        except SkipModel as e:
            last_error = e
            log(f"Modello {model} saltato, provo il successivo.")
    raise RuntimeError(f"Tutti i modelli hanno fallito. Ultimo errore: {last_error}")


def clean_html(raw):
    fence = "`" * 3
    content = raw
    if fence + "html" in content:
        content = content.split(fence + "html")[-1].split(fence)[0]
    elif fence in content:
        content = content.split(fence)[1]
    return content.strip()


def main():
    html_content = clean_html(generate_with_fallback())

    # Validazione: non committare mai un file incompleto
    count = html_content.count('class="notification"')
    if "<html" not in html_content or count != EXPECTED_NOTIFICATIONS:
        raise ValueError(
            f"HTML non valido: trovate {count} notifiche su {EXPECTED_NOTIFICATIONS}.\n"
            f"Anteprima output:\n{html_content[:500]}"
        )

    # Sostituzione ID univoco
    html_content = re.sub(r'id="[^"]+"', f'id="{unique_id}"', html_content)

    # Salvataggio
    folder_path = "interact"
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "push_notifications.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    log(f"File aggiornato con successo. Argomento: {current_focus}")


if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
        os._exit(0)
    except Exception as e:
        log(f"Errore critico finale: {e}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
