import requests
import sys

# API URL — must match the port uvicorn is running on
API_URL = "http://127.0.0.1:8003/predict"


def ask_ai():
    print("--------------------------------------------------")
    print("🤖 Indian Market AI: Ask Me Anything (Type 'exit' to quit)")
    print("--------------------------------------------------")
    
    while True:
        symbol = input("\n📈 Enter Stock Symbol (e.g. TCS, INFY): ").strip().upper()
        
        if symbol.lower() == 'exit':
            print("Goodbye! 👋")
            break
            
        if not symbol:
            continue
            
        if not symbol.endswith(".NS"):
            symbol += ".NS"
            
        try:
            print(f"thinking about {symbol}...")
            response = requests.post(API_URL, json={"symbol": symbol})
            
            if response.status_code == 200:
                data = response.json()
                prob = data['probability'] * 100
                signal = "🟢 STRONG BUY" if data['prediction'] == 'Buy' and prob > 60 else \
                         "🟢 WEAK BUY" if data['prediction'] == 'Buy' else \
                         "🔴 SELL"
                
                print(f"\n👉 Recommendation: {signal}")
                print(f"📊 Probability:    {prob:.1f}%")
                print(f"📉 RSI (14):       {data['rsi']:.1f}")
                print(f"📰 Sentiment:      {data['sentiment']:.2f}")
                print(f"📝 Model Note:     {data['note']}")
            elif response.status_code == 404:
                print("❌ Error: Stock data not found in Feature Store. (Did you run feature_store.py today?)")
            else:
                print(f"❌ Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error connecting to AI: {e}")

if __name__ == "__main__":
    ask_ai()
