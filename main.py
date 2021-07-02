## text preprocessing modules
from string import punctuation
## text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  ## regular expression
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews "
                "or tu user NER on a giver phrase, "
                "or even to classify given SMS messages as SPAM or not",
    version="0.1",
)

## load the sentiment model
with open(
    join(dirname(realpath(__file__)), "sentiment_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    ## Clean the text, with the option to remove stop_words and to lemmatize word
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  ## remove numbers
    ## Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    ## Optionally, remove stop words
    if remove_stop_words:
        ## load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    ## Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    ## Return a list of words
    return text

@app.get("/predict-review")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    ## clean the review
    cleaned_review = text_cleaning(review)

    ## perform prediction
    prediction = model.predict([cleaned_review])
    output = int(prediction[0])
    probas = model.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probas[:, output]))

    ## output dictionary
    sentiments = {0: "Negative", 1: "Positive"}

    ## show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result

@app.get("/preidct-ner")
def predict_ner(phrase = "My name is Sophie I am from America I want to work with Google Steve Jobs is My Inspiration"):
    import spacy
    from spacy import displacy

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(phrase)
    a = []
    b = []
    c = []
    d = []
    e = {}

    for chunk in doc.noun_chunks:
        a.append(chunk.text)

    for token in doc:
        if token.pos_ == "VERB":
            b.append(token.lemma_)

    for entity in doc.ents:
       c.append(doc.ents)
       d.append(entity.label_)

    for key in c:
        for value in d:
            e[key]=value
            d.remove(value)
            break

    res = {"Nouns:", str(a), "Verbs:", str(b), "Entities:", str(e)}

    return res

@app.get("/predict-sms")
def predict_sms(sms="You have an announcement. Call FREEPHONE 0125 2555 011 now!"):
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # Features and Labels
    df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
    X = df['v2']
    y = df['label']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier


    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    # Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)


    data = [sms]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)

    if my_prediction == 1:
        return "It is a SPAM"
    elif my_prediction == 0:
        return "It is a HAM"
    else:
        return "There was a mistake parsing your request, retry please"