from fastapi import FastAPI
from sentiment import Sentiment
from ner import NER
from classifier import Classifier

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews "
                "or tu user NER on a giver phrase, "
                "or even to classify given SMS messages as SPAM or not",
    version="0.1",
)


@app.get("/predict-review")
def predict_sentiment(review: str):
    S = Sentiment()
    return S.predict(review)


@app.get("/preidct-ner")
def predict_ner(phrase="My name is Sophie I am from America I want to work with Google Steve Jobs is My Inspiration"):
    ner = NER()
    return ner.predict(phrase)


@app.get("/predict-sms")
def predict_sms(sms="You have an announcement. Call FREEPHONE 0125 2555 011 now!"):
    C = Classifier()
    return C.predict(sms)
