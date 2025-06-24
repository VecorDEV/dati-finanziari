import os
import pandas as pd
import requests
import csv

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

def filtra_e_calcola_sentiment(df, keyword, max_links=50):
    mask_link = df['DocumentIdentifier'].str.contains(keyword, case=False, na=False)
    mask_theme = df['V2Themes'].str.contains(keyword, case=False, na=False)
    filtered = df[mask_link | mask_theme]
    
    if filtered.empty:
        return None, None, None
    
    tones = filtered['V2Tone'].dropna().apply(lambda x: float(str(x).split(',')[0]))
    sentiment_mean = tones.mean() if not tones.empty else None

    links = filtered['DocumentIdentifier'].dropna().unique()[:max_links]
    return sentiment_mean, len(links), ";".join(links)

def aggiorna_file_gdelt():
    file_url = scarica_ultimo_gdelt_file()
    scarica_file_localmente(file_url, GDELT_LOCAL_FILE)

if __name__ == "__main__":
    if not os.path.exists(GDELT_LOCAL_FILE):
        print("File GDELT locale non trovato, scaricamento automatico...")
        aggiorna_file_gdelt()

    df = carica_gdelt_csv(GDELT_LOCAL_FILE)
    
    symbols = ["AAPL", "BTC", "GOLD", "GOOG", "TSLA", "ETH"]  # <-- Sostituisci con i tuoi 120 simboli

    risultati = []

    for asset in symbols:
        sentiment, num_links, links = filtra_e_calcola_sentiment(df, asset)
        if sentiment is not None:
            print(f"{asset} -> Sentiment medio: {sentiment:.2f} su {num_links} articoli.")
            risultati.append([asset, sentiment, num_links, links])
        else:
            print(f"{asset} -> Nessun dato trovato.")
            risultati.append([asset, "N/D", 0, ""])

    output_file = "sentiment_risultati.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Asset", "Sentiment medio", "Numero articoli", "Lista URL articoli"])
        writer.writerows(risultati)

    print(f"\nRisultati salvati in '{output_file}'")
