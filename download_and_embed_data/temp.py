import requests

url = "http://localhost:6333/collections/chunks/points/scroll"
payload = {"limit": 10, "with_payload": True, "with_vector": True}

r = requests.post(url, json=payload)

data = r.json()
points = data["result"]["points"]
print(data)
for p in points:
    text_len = len(p["payload"]["text"])
    vector_len = len(p["vector"])
    print(text_len, vector_len)