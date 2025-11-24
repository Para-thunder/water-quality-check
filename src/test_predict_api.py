import json
import urllib.request

url = 'http://127.0.0.1:8000/predict'
headers = {'Content-Type': 'application/json'}
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
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
with urllib.request.urlopen(req, timeout=10) as resp:
    print(resp.status)
    print(resp.read().decode('utf-8'))
