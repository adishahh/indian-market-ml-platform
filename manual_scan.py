import requests
import json

symbols = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS"]
url = "http://127.0.0.1:8004/predict"

print(f"--- Running Manual Scan (API: {url}) ---\n")

for symbol in symbols:
    try:
        response = requests.post(url, json={"symbol": symbol})
        if response.status_code == 200:
            data = response.json()
            # Format nicely
            prob = data['probability'] * 100
            signal = "🟢 BUY" if data['prediction'] == 'Buy' else "🔴 SELL"
            
            print(f"{symbol}: {signal} (Prob: {prob:.1f}%)")
            print(f"  RSI: {data['rsi']:.1f}, Sentiment: {data['sentiment']:.2f}")
            print(f"  Note: {data['note']}\n")
        else:
            print(f"{symbol}: Error {response.status_code} - {response.text}")
    except Exception as e:
        print(f"{symbol}: Failed - {str(e)}")
