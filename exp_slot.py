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
from eva import evaluate, save
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

numpy.set_printoptions(threshold=sys.maxsize)
params = {
    "cat":  ('comp.graphics', 'rec.autos', 'rec.sport.baseball',), 
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
    vectorizer = TfidfVectorizer(norm=None, stop_words='english', ngram_range=(1,3), max_features=params['num_slots'])
    vectorizer.fit(train.data)
    print("Vectorizer fitting done. Vocabulary size: " + str(len(vectorizer.vocabulary_)))

    train_vectors = vectorizer.transform(train.data)
    test_vectors = vectorizer.transform(test.data)

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
    return numpy.squeeze(numpy.asarray(vectorizer.transform(test.data).todense())).tolist(), test.target

def run_eval(i: int, public_ctx, secret_ctx, inputs, outputs, signature, compiled_prog):
    print(i)
    enc_inputs = public_ctx.encrypt({'x' : inputs[i]}, signature)
    enc_outputs = public_ctx.execute(compiled_prog, enc_inputs)
    outputs[i] = secret_ctx.decrypt(enc_outputs, signature)

def compile_run(program, inputs):
    compiler = CKKSCompiler()
    print("Compiling SEAL program!")
    previous_time = time.time()
    compiled_prog, params, signature = compiler.compile(program)
    print("--- %s seconds ---" % (time.time() - previous_time))
    
    print("Generating encryption!")
    previous_time = time.time()
    public_ctx, secret_ctx = generate_keys(params)
    print("--- %s seconds ---" % (time.time() - previous_time))
    save(public_ctx, "public.test")
    enc_inputs, enc_outputs, outputs = [[None] * len(inputs)] * 3
    print(len(inputs))
    
    # print("Encoding!")
    # previous_time = time.time()
    # for i in range(len(inputs)):
    #     if i % 100 == 0:
    #         printProgressBar(i // 100, len(inputs) // 100)
    #     enc_inputs[i] = public_ctx.encrypt({'x' : inputs[i]}, signature)
    # print("--- %s seconds ---" % (time.time() - previous_time))
        

    # print("Executing!")
    # previous_time = time.time()
    # for i in range(len(inputs)):
    #     if i % 100 == 0:
    #         printProgressBar(i // 100, len(inputs) // 100)
    #     enc_outputs[i] = public_ctx.execute(compiled_prog, enc_inputs[i])
    # print("--- %s seconds ---" % (time.time() - previous_time))


    # print("Decoding!")
    # previous_time = time.time()
    # for i in range(len(inputs)):
    #     if i % 100 == 0:
    #         printProgressBar(i // 100, len(inputs) // 100)
    #     outputs[i] = secret_ctx.decrypt(enc_outputs[i], signature)
    # print("--- %s seconds ---" % (time.time() - previous_time))
    # return outputs


def horizontal_sum(x):
	i = 1
	while i < x.program.vec_size:
		y = x << i
		x = x + y
		i <<= 1
	return x

def create_eva_program(num_cat: int, idf_weights: List[float], coefficients: List[List[float]], intercept: List[float]):
    encrypted_svc = EvaProgram('resolve_svc', vec_size=params['num_slots'])
    
    with encrypted_svc:
        x = Input('x')
        x = x * idf_weights

        for category in range(num_cat):
            x_weighted = x * coefficients[category]
            x_summed = horizontal_sum(x_weighted)
            x_intercept = x_summed + intercept[category]
            Output("x{}".format(category), x_intercept)

    encrypted_svc.set_input_scales(40)
    encrypted_svc.set_output_ranges(30)

    return encrypted_svc

def main():
    vectorizer, classifier = train_model()
    program = create_eva_program(3, vectorizer.idf_.tolist(), classifier.coef_.tolist(), classifier.intercept_.tolist())
    samples, targets = generate_test_inputs(vectorizer.vocabulary_)
    outputs = compile_run(program, samples)
    
    scores = []
    for i in range(len(samples)):
        scores.append([])
        for key in sorted(outputs[0].keys()):
            scores[i].append(outputs[i][key][0])
    classes = numpy.array([0, 1, 2,])
    indices = numpy.array(scores).argmax(axis=1)
    f1 = metrics.f1_score(targets, classes[indices], average='weighted')
    print(f1)
            

if __name__ == "__main__":
    param_list = [4096]
    for param in param_list:
        print("\n\n\n")
        params["num_slots"] = param
        main()