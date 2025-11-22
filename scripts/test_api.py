import requests
import json

url = "http://127.0.0.1:8000/predict"

# Sample data for a water sample
data = {
    "ph": 7.0,
    "Hardness": 200.0,
    "Solids": 20000.0,
    "Chloramines": 7.0,
    "Sulfate": 300.0,
    "Conductivity": 400.0,
    "Organic_carbon": 15.0,
    "Trihalomethanes": 60.0,
    "Turbidity": 4.0
}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("\n✅ Success! API Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"\n❌ Failed to connect: {e}")
    print("Make sure the API is running (uvicorn command).")
