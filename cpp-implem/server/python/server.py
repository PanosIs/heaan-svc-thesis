import json
import re

from flask import request
from sklearn import datasets, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
import flask
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    train = datasets.fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        categories=('sci.med', 'sci.space', 'sci.electronics')
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(norm=None, stop_words='english', max_features=8192, ngram_range=(1, 3))
    vectorizer.fit(train.data)
    print("Vectorizer fitting done. Vocabulary size: " + str(len(vectorizer.vocabulary_)))

    train_vectors = vectorizer.transform(train.data)

    print("Fitting support vector classifier model...")
    classifier = LinearSVC(C=0.05, loss='squared_hinge')
    classifier.fit(train_vectors, train.target)
    prediction = classifier.predict(train_vectors)
    print("Model training done. Training accuracy: " + str(metrics.f1_score(train.target, prediction, average='weighted')))
    test = datasets.fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        categories=('sci.med', 'sci.space', 'sci.electronics')
    )
    prediction = classifier.predict(vectorizer.transform(test.data))
    print("Test accuracy: " + str(metrics.f1_score(test.target, prediction, average='weighted')))


    app = flask.Flask(__name__)

    # This route is used by the client to vectorize the samples before encryption
    @app.route('/vocabulary', methods=['GET'])
    def get_vocabulary():
        print("Providing vocabulary list to " + str(request.remote_addr))
        print("Vocabulary size: " + str(len(vectorizer.vocabulary_)))
        return str(vectorizer.vocabulary_).replace("\'", "\"")

    # This route is used by the server to retrieve model parameters
    @app.route('/model', methods=['GET'])
    def get_samples():
        print("Providing model parameters to " + str(request.remote_addr))
        params = dict()
        params['coefficients'] = str(classifier.coef_.tolist()).replace("], [", "];[").replace(", ", ",")
        params['intercept'] = str(classifier.intercept_.tolist()).replace(", ", ",")

        return str(params)

    # This route is used by the server to retrieve the IDF factors
    @app.route('/idf', methods=['GET'])
    def get_idf():
        print("Providing idf weights to " + str(request.remote_addr))
        return re.sub(r' +', ' ', str(vectorizer.idf_)\
            .replace("[", "")\
            .replace("]","")\
            .replace("\n",""))

    app.run()
