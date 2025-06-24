import os
import pandas as pd
import requests

GDELT_LOCAL_FILE = "gdelt_latest.csv"

def scarica_ultimo_gdelt_file():
    url = 'http://data.gdeltproject.org/gdeltv2/lastupdate.txt'
    r = requests.get(url)
    r.raise_for_status()
    last_update_info = r.text.strip().split('\n')[0]
    file_name = last_update_info.split(' ')[-1]
    file_url = f"http://data.gdeltproject.org/gdeltv2/{file_name}"
    return file_url

def scarica_file_localmente(file_url, local_file):
    print("Controllo file GDELT locale...")
    if os.path.exists(local_file):
        print(f"File locale trovato: {local_file}")
        return local_file
    print("Scaricamento file GDELT...")
    r = requests.get(file_url, stream=True)
    with open(local_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("File scaricato.")
    return local_file

def carica_gdelt_csv(local_file):
    columns = [
        "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName", "DocumentIdentifier", 
        "Counts", "V2Counts", "Themes", "V2Themes", "Locations", "V2Locations", 
        "Persons", "V2Persons", "Organizations", "V2Organizations", "V2Tone", 
        "Dates", "GCAM", "SharingImage", "RelatedImages", "SocialImageEmbeds", 
        "SocialVideoEmbeds", "Quotations", "AllNames", "Amounts", "TranslationInfo", 
        "Extras"
    ]
    df = pd.read_csv(local_file, sep='\t', names=columns, quoting=3, low_memory=False)
    return df

def filtra_e_calcola_sentiment(df, keyword, max_links=5):
    mask_link = df['DocumentIdentifier'].str.contains(keyword, case=False, na=False)
    mask_theme = df['V2Themes'].str.contains(keyword, case=False, na=False)
    filtered = df[mask_link | mask_theme]
    
    if filtered.empty:
        return None, 0, []

    tones = filtered['V2Tone'].dropna().apply(lambda x: float(str(x).split(',')[0]))
    sentiment_mean = tones.mean() if not tones.empty else None

    links = filtered['DocumentIdentifier'].dropna().unique()[:max_links]
    return sentiment_mean, len(links), list(links)

def aggiorna_file_gdelt():
    file_url = scarica_ultimo_gdelt_file()
    scarica_file_localmente(file_url, GDELT_LOCAL_FILE)

if __name__ == "__main__":
    aggiorna_file_gdelt()
    df = carica_gdelt_csv(GDELT_LOCAL_FILE)
    
    symbols = ["AAPL", "BTC", "GOLD", "GOOG", "TSLA", "ETH"]  # <-- Sostituisci con la tua lista

    for asset in symbols:
        sentiment, num_links, links = filtra_e_calcola_sentiment(df, asset)
        print(f"\n--- {asset} ---")
        if sentiment is not None:
            print(f"Sentiment medio: {sentiment:.2f}")
            print(f"Numero articoli: {num_links}")
            print("URL articoli (max 5):")
            for url in links:
                print(f" - {url}")
        else:
            print("Nessun dato trovato.") 
