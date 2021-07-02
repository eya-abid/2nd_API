import requests as r

## add review
phrase = "Hi, My name is Sophie I am from America I want to work with Google Steve Jobs is My Inspiration"

keys = {"review": phrase}
prediction = r.get("http://127.0.0.1:8000/predict-ner/", params=keys)
results = prediction.json()

print(results["prediction"])