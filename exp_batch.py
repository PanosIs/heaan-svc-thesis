import json
import re
from typing import List

from flask import request
from sklearn import datasets, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
import flask
import sys
import numpy
import math
import time
from eva import EvaProgram, Input, Output
import unittest
from random import random, uniform
from eva import evaluate
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse

numpy.set_printoptions(threshold=sys.maxsize)
params = {
    "cat":  ('comp.graphics', 'rec.autos', 'rec.sport.baseball', 'sci.crypt', 'talk.politics.mideast', 'soc.religion.christian',  'talk.politics.guns', 'rec.motorcycles',), 
    "num_params": 128, 
    "num_slots": 128
    }


def train_model():
    train = datasets.fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        categories=params['cat'],
        shuffle=False
    )
    test = datasets.fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        categories=params['cat'],
        shuffle=False
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=params['num_params'])
    vectorizer.fit(train.data)
    print("Vectorizer fitting done. Vocabulary size: " + str(len(vectorizer.vocabulary_)))

    train_vectors = vectorizer.transform(train.data)
    test_vectors = vectorizer.transform(test.data)
    print(test_vectors.shape)

    print("Fitting support vector classifier model...")
    classifier = LinearSVC(C=0.00025, class_weight='balanced')
    classifier.fit(train_vectors, train.target)
    prediction = classifier.predict(train_vectors)
    print("Model training done. Training accuracy: " + str(metrics.f1_score(train.target, prediction, average='weighted')))
    prediction = classifier.predict(test_vectors)
    print("Test accuracy: " + str(metrics.f1_score(test.target, prediction, average='weighted')))

    return vectorizer, classifier

def generate_test_inputs(vocabulary):
    vectorizer = CountVectorizer(ngram_range=(1,3))
    test = datasets.fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        categories=params['cat'],
        shuffle=False
    )
    vectorizer.vocabulary_ = vocabulary
    return numpy.squeeze(numpy.asarray(vectorizer.transform(test.data).todense())), test.target

def process_inputs(samples):
    processed_inputs = {}
    for i in range(params['num_params']):
        processed_inputs[f'x{i}'] = []

    for i in range(params['num_params']):
        for j in range(params['num_slots']):
            if j < len(samples):
                processed_inputs[f'x{i}'].append(samples[j][i])    
            else:
                processed_inputs[f'x{i}'].append(0)

    return processed_inputs


def compile_run(program, inputs):
    compiler = CKKSCompiler(config={'warn_vec_size':'false'})
    previous_time = time.time()
    print("Compiling SEAL program!")
    compiled_prog, params, signature = compiler.compile(program)
    print("--- %s seconds ---" % (time.time() - previous_time))
    with open('out.txt', 'w') as f:
        print(compiled_prog.to_DOT(), file=f)

    print("Generating encryption!")
    previous_time = time.time()
    public_ctx, secret_ctx = generate_keys(params)
    print("--- %s seconds ---" % (time.time() - previous_time))
    print("Encoding!")
    previous_time = time.time()
    enc_inputs = public_ctx.encrypt(inputs, signature)
    print("--- %s seconds ---" % (time.time() - previous_time))
    print("Executing!")
    previous_time = time.time()
    enc_outputs = public_ctx.execute(compiled_prog, enc_inputs)
    print("--- %s seconds ---" % (time.time() - previous_time))
    print("Decrypting!")
    previous_time = time.time()
    outputs = secret_ctx.decrypt(enc_outputs, signature)
    reference = evaluate(compiled_prog, inputs)
    print("--- %s seconds ---" % (time.time() - previous_time))
    print('MSE', valuation_mse(outputs, reference))
    return outputs


def create_eva_program(num_cat: int, idf_weights: List[float], coefficients, intercept):
    num_params = params['num_params']
    encrypted_svc = EvaProgram('resolve_svc', vec_size=params['num_slots'])
    
    with encrypted_svc:
        inputs = [Input(f'x{i}') for i in range(num_params)]
        
        x_idf = [input * constant for input, constant in zip(inputs, idf_weights)]

        for category in range(num_cat):
            x_weighted = [x * weight for x, weight in zip(x_idf, coefficients[category])]
            x_summed = sum(x_weighted)
            x_intercept = x_summed + intercept[category]
            Output(f'c{category}', x_intercept)

    encrypted_svc.set_input_scales(40)
    encrypted_svc.set_output_ranges(30)

    return encrypted_svc

def main():
    print(params)
    vectorizer, classifier = train_model()
    program = create_eva_program(8, vectorizer.idf_.tolist(), classifier.coef_.tolist(), classifier.intercept_.tolist())
    samples, targets = generate_test_inputs(vectorizer.vocabulary_)
    processed_inputs = process_inputs(samples)
    outputs = compile_run(program, processed_inputs)

    scores = []
    print(samples.shape[0])
    for i in range(samples.shape[0]):
        scores.append([])
        for key in sorted(outputs.keys()):
            scores[i].append(outputs[key][i])
    classes = numpy.array([0, 1, 2, 3, 4, 5, 6, 7])
    indices = numpy.array(scores).argmax(axis=1)
    f1 = metrics.f1_score(targets, classes[indices], average='weighted')
    print(f1)
            

if __name__ == "__main__":
    param_list = [4096]
    for param in param_list:
        print("\n\n\n")
        params["num_params"] = param
        main()