import requests as r

## add review
sms = "You have an announcement. Call FREEPHONE 0125 2555 011 now!"

keys = {"review": sms}
prediction = r.get("http://127.0.0.1:8000/predict-sms/", params=keys)
results = prediction.json()

print(results["prediction"])