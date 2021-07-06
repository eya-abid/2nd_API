from abstract import Abstract
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class Classifier(Abstract):

    def predict(self, text: str):
        df = pd.read_csv("spam.csv", encoding="latin-1")
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
        X = df['v2']
        y = df['label']

        cv = CountVectorizer()
        X = cv.fit_transform(X)  # Fit the Data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        clf.score(X_test, y_test)

        data = [text]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

        if my_prediction == 1:
            return "It is a SPAM"
        elif my_prediction == 0:
            return "It is a HAM"
        else:
            return "There was a mistake parsing your request, retry please"
