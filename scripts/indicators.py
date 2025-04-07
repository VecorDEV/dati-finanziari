import requests
import datetime
import os

API_KEY = '186ca58fa08b4551ad7950f24491ef96'
ASSETS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # Puoi ampliarla fino a 100 asset
INTERVAL = '1day'
BASE_URL = 'https://api.twelvedata.com'
OUTPUT_FILE = 'result/indicatori.html'

def get_indicator(symbol, indicator, params=""):
    url = f"{BASE_URL}/{indicator}?symbol={symbol}&interval={INTERVAL}&apikey={API_KEY}{params}"
    response = requests.get(url)
    return response.json()

def calcola_probabilita(rsi, macd_cross, ema_cross):
    punteggio = 0

    if rsi > 70:
        punteggio -= 10
    elif rsi < 30:
        punteggio += 10

    if macd_cross == "rialzista":
        punteggio += 15
    elif macd_cross == "ribassista":
        punteggio -= 15

    if ema_cross == "rialzista":
        punteggio += 10
    elif ema_cross == "ribassista":
        punteggio -= 10

    probabilita = (punteggio + 100) / 2
    return round(probabilita, 2)

def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    os.makedirs("result", exist_ok=True)
    
    html = f"<html><head><title>Indicatori Tecnici - {now}</title></head><body>"
    html += f"<h1>Indicatori tecnici aggiornati al {now}</h1>"

    for symbol in ASSETS:
        try:
            rsi_data = get_indicator(symbol, 'rsi', '&time_period=14')
            macd_data = get_indicator(symbol, 'macd')
            ema_9_data = get_indicator(symbol, 'ema', '&time_period=9')
            ema_21_data = get_indicator(symbol, 'ema', '&time_period=21')

            rsi = float(rsi_data['values'][0]['rsi']) if 'values' in rsi_data else None
            macd = float(macd_data['values'][0]['macd']) if 'values' in macd_data else None
            signal = float(macd_data['values'][0]['signal']) if 'values' in macd_data else None
            ema_9 = float(ema_9_data['values'][0]['ema']) if 'values' in ema_9_data else None
            ema_21 = float(ema_21_data['values'][0]['ema']) if 'values' in ema_21_data else None

            macd_cross = "rialzista" if macd > signal else "ribassista"
            ema_cross = "rialzista" if ema_9 > ema_21 else "ribassista"
            probabilita = calcola_probabilita(rsi, macd_cross, ema_cross)

            html += f"""
                <h2>{symbol}</h2>
                <ul>
                    <li><strong>RSI (14):</strong> {rsi}</li>
                    <li><strong>MACD:</strong> {macd}</li>
                    <li><strong>Signal Line:</strong> {signal}</li>
                    <li><strong>EMA 9:</strong> {ema_9}</li>
                    <li><strong>EMA 21:</strong> {ema_21}</li>
                    <li><strong>MACD Trend:</strong> {macd_cross}</li>
                    <li><strong>EMA Trend:</strong> {ema_cross}</li>
                    <li><strong>Probabilità di crescita:</strong> {probabilita}%</li>
                </ul>
                <hr>
            """
        except Exception as e:
            html += f"<h2>{symbol}</h2><p>Errore: {str(e)}</p><hr>"

    html += "</body></html>"

    with open(OUTPUT_FILE, 'w') as f:
        f.write(html)

if __name__ == '__main__':
    main()
