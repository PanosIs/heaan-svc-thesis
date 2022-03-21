import json

import re
import requests
from flask import request
from sklearn import datasets, metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import flask
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)


class Client:
    def __init__(self):
        self.vectorizer = CountVectorizer(ngram_range=(1,3))
        self.classes = None
        self.samples = None

    def set_samples(self, samples):
        self.samples = samples

    def set_vocabulary(self, vocab):
        print("Setting vocabulary for client vectorizer...")
        self.vectorizer.vocabulary_ = vocab
        print("Vocabulary size: " + str(len(self.vectorizer.vocabulary_)))

    def get_vectors(self):
        print("Client providing vectorized samples...")
        print("Total samples: " + str(len(self.samples.data)))
        if (self.samples is not None) and (self.vectorizer.vocabulary_ is not None):
            return self.vectorizer.transform(self.samples.data)
        else:
            raise Exception("Vocabulary or samples have not been set on the client.")

    def provide_predictions(self, scores):
        classes = numpy.array([0, 1, 2])
        print("Client receiving predictions and classes...")
        print("Classes: " + str(classes))
        print(scores)
        print(len(scores))
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        f1 = metrics.f1_score(self.samples.target[:len(scores)], classes[indices], average='weighted')
        print("Accuracy: " + str(f1))
        return f1

def parse_predict(vector):
    vstring = re.sub(r' +', ',', str(vector.todense())\
        .replace("[", "")\
        .replace("]","")\
        .replace("\n","")).rstrip(',').lstrip(',')
    resp = requests.post("http://127.0.0.1:9081/score", vstring).text
    try:
        return [float(x) for x in resp.split(",")]
    except Exception:
        print(vstring)
        print(resp)
        return [0, 0, 0]

if __name__ == "__main__":
    client = Client()

    test = datasets.fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        categories=('sci.med', 'sci.space', 'sci.electronics')
    )

    client.set_samples(test)

    vocab = requests.get("http://127.0.0.1:5000/vocabulary")
    json_vocab = json.loads(vocab.text)
    client.set_vocabulary(json_vocab)

    vectors = client.get_vectors()
    parsed = [parse_predict(v) for v in vectors]
    client.provide_predictions(numpy.array(parsed))

