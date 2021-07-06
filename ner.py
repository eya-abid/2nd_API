from abstract import Abstract
import spacy


class NER(Abstract):

    def predict(self, text: str):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        a = []
        b = []
        c = []
        d = []
        for chunk in doc.noun_chunks:
            a.append(chunk.text)

        for token in doc:
            if token.pos_ == "VERB":
                b.append(token.lemma_)

        for entity in doc.ents:
            c.append(entity.ents)
            d.append(entity.label_)

        e = dict(zip(d, c))
        res = "Nouns:" + str(a) + " , Verbs:" + str(b) + " , Entities:" + str(e)
        return res
