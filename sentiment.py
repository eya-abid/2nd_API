from abstract import Abstract
from string import punctuation
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from os.path import dirname, join, realpath
import joblib

# load the sentiment model
with open(
    join(dirname(realpath(__file__)), "sentiment_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)


class Sentiment(Abstract):

    def text_cleaning(self, text, remove_stop_words=True, lemmatize_words=True):
        # Clean the text, with the option to remove stop_words and to lemmatize word
        # Clean the text
        text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"http\S+", " link ", text)
        text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
        # Remove punctuation from text
        text = "".join([c for c in text if c not in punctuation])
        # Optionally, remove stop words
        if remove_stop_words:
            # load stopwords
            stop_words = stopwords.words("english")
            text = text.split()
            text = [w for w in text if not w in stop_words]
            text = " ".join(text)
        # Optionally, shorten words to their stems
        if lemmatize_words:
            text = text.split()
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
            text = " ".join(lemmatized_words)
        # Return a list of words
        return text

    def predict(self, review):
        cleaned_review = self.text_cleaning(review)
        prediction = model.predict([cleaned_review])
        output = int(prediction[0])
        probas = model.predict_proba([cleaned_review])
        output_probability = "{:.2f}".format(float(probas[:, output]))
        sentiments = {0: "Negative", 1: "Positive"}
        result = {"prediction": sentiments[output], "Probability": output_probability}
        return result
