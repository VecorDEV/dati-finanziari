import yfinance as yf
import ta

# Scarica i dati di TSLA
tsla_data = yf.download("TSLA", period="1y", interval="1d")

# Calcola RSI (Relative Strength Index)
tsla_data['rsi'] = ta.momentum.RSIIndicator(close=tsla_data['Close']).rsi()

# Calcola MACD (Moving Average Convergence Divergence)
tsla_data['macd'] = ta.trend.MACD(close=tsla_data['Close']).macd()
tsla_data['macd_signal'] = ta.trend.MACD(close=tsla_data['Close']).macd_signal()

# Calcola medie mobili (SMA 20 e EMA 20)
tsla_data['sma_20'] = ta.trend.SMAIndicator(close=tsla_data['Close'], window=20).sma_indicator()
tsla_data['ema_20'] = ta.trend.EMAIndicator(close=tsla_data['Close'], window=20).ema_indicator()

# Mostra le ultime righe con gli indicatori
print(tsla_data[['Close', 'rsi', 'macd', 'macd_signal', 'sma_20', 'ema_20']].tail())
