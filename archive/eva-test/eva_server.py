import time
import uuid
import os
import io
from typing import List
from eva import EvaProgram, Input, Output
from eva import evaluate, save, load
from eva.ckks import CKKSCompiler
from flask import Flask, send_file
from flask import request
from sklearn import datasets, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import json
import requests
from sklearn.feature_extraction.text import CountVectorizer
from eva import load, save
from eva.seal import generate_keys
import numpy
from sklearn import datasets
import string
import random

alphabet = string.ascii_lowercase + string.digits

class EvaWrapper:
    def __init__(self, name: str) -> None:
        self.name = name
        self.create_program()

    def create_program(self):
        print('Compile time')
        poly = EvaProgram('Polynomial', vec_size=8)
        with poly:
            x = Input('x')
            Output('y', 3*x**2 + 5*x - 2)

        poly.set_output_ranges(20)
        poly.set_input_scales(20)

        compiler = CKKSCompiler()
        poly, params, signature = compiler.compile(poly)

        save(poly, f'{self.name}.eva')
        save(params, f'{self.name}.evaparams')
        save(signature, f'{self.name}.evasignature')
    
    def execute(self):
        print('Runtime on server')
        program = load(f'{self.name}.eva')
        input = load(f'{self.name}.inputs.sealvals')
        ctx = load(f'{self.name}.sealpublic')
        output = ctx.execute(program, input)
        save(output, f'{self.name}.outputs.sealvals')


app = Flask(__name__)
eva = EvaWrapper('poly')

@app.route('/signature', methods=['GET'])
def get_signature():
    return send_file(f'{eva.name}.evasignature', as_attachment=True)

@app.route('/params', methods=['GET'])
def get_params():
    return send_file(f'{eva.name}.evaparams', as_attachment=True)

@app.route('/public', methods=['POST'])
def set_session_context():
    if 'file' not in request.files:
        return {"error": "No context file provided."}, 406

    file = request.files['file']
    file.save(f'{eva.name}.sealpublic')
    return {"status": "Context loaded."}, 200

@app.route('/evaluate', methods=['POST'])
def set_session_input():
    if 'file' not in request.files:
        return {"error": "No context file provided."}, 406

    file = request.files['file']
    file.save(f'{eva.name}.inputs.sealvals')
    eva.execute()
    return send_file(f'{eva.name}.outputs.sealvals', as_attachment=True)

@app.route('/localrun', methods=['POST'])
def localrun():
    client_process()
    return {}, 200

def client_process():
    #####################################################
    print('Key generation time')
    params = load('poly.evaparams')

    public_ctx, secret_ctx = generate_keys(params)

    save(public_ctx, 'poly.sealpublic')
    save(secret_ctx, 'poly.sealsecret')
    
    #####################################################
    print('Runtime on client')
    signature = load('poly.evasignature')
    public_ctx = load('poly.sealpublic')

    inputs = {
        'x': [i for i in range(signature.vec_size)]
    }
    print('Encrypting')
    encInputs = public_ctx.encrypt(inputs, signature)

    save(encInputs, 'poly.input.sealvals')
    #####################################################
    eva.execute()
    #####################################################

if __name__ == "__main__":
    app.run(threaded=False)

# CATEGORIES = ('comp.graphics', 'rec.autos', 'rec.sport.baseball', 'sci.crypt', 'talk.politics.mideast', 'soc.religion.christian',  'talk.politics.guns', 'rec.motorcycles',)
# NUM_FEATURES = 4096

# # Eva Application Wrapper

# class EvaApp:
#     def __init__(self) -> None:
#         self.sessions = {}
#         self.vectorizer = None
#         self.classifier = None
#         self.train_model()
#         self.create_eva_program()

#     def new_session(self) -> str:
#         session = EvaSession(self)
#         self.sessions[session.uuid] = session
#         return session.uuid

#     def train_model(self) -> None:
#         train = datasets.fetch_20newsgroups(
#             subset='train',
#             remove=('headers', 'footers', 'quotes'),
#             categories=CATEGORIES,
#             shuffle=False
#         )
        
#         test = datasets.fetch_20newsgroups(
#             subset='test',
#             remove=('headers', 'footers', 'quotes'),
#             categories=CATEGORIES,
#             shuffle=False
#         )

#         print("Fitting TF-IDF vectorizer...")
#         vectorizer = TfidfVectorizer(norm=None, stop_words='english', ngram_range=(1,3), max_features=NUM_FEATURES)
#         vectorizer.fit(train.data)
#         print("Vectorizer fitting done. Vocabulary size: " + str(len(vectorizer.vocabulary_)))

#         train_vectors = vectorizer.transform(train.data)
#         test_vectors = vectorizer.transform(test.data)

#         print("Fitting support vector classifier model...")
#         classifier = LinearSVC(C=0.00025, class_weight='balanced')
#         classifier.fit(train_vectors, train.target)
#         prediction = classifier.predict(train_vectors)
#         print("Model training done. Training accuracy: " + str(metrics.f1_score(train.target, prediction, average='weighted')))
#         prediction = classifier.predict(test_vectors)
#         print("Test accuracy: " + str(metrics.f1_score(test.target, prediction, average='weighted')))

#         self.vectorizer = vectorizer
#         self.classifier = classifier

#     def create_eva_program(self):
#         if self.vectorizer is None or self.classifier is None: 
#             print("No unencrypted model registered for encryption, no action taken...")
#             return

#         encrypted_svc = EvaProgram('resolve_svc', vec_size=NUM_FEATURES)
        
#         with encrypted_svc:
#             x = Input('x')
#             x = x * self.vectorizer.idf_.tolist()

#             for category in range(len(CATEGORIES)):
#                 x_weighted = x * self.classifier.coef_.tolist()[category]
#                 x_summed = self.horizontal_sum(x_weighted)
#                 x_intercept = x_summed + self.classifier.intercept_.tolist()[category]
#                 Output("x{}".format(category), x_intercept)

#         encrypted_svc.set_input_scales(40)
#         encrypted_svc.set_output_ranges(30)

#         self.program = encrypted_svc

#     @staticmethod
#     def horizontal_sum(x):
#         i = 1
#         while i < x.program.vec_size:
#             y = x << i
#             x = x + y
#             i <<= 1
#         return x
        
# # Eva Session Object

# class EvaSession:
#     def __init__(self, parent: EvaApp) -> None:
#         self.state = 0
#         self.parent = parent
#         self.uuid = uuid.uuid4().hex
#         self.public_ctx_path = f"session.{self.uuid}.sealpublic"
#         self.public_ctx = None

#         compiler = CKKSCompiler()
#         compiled_prog, params, signature = compiler.compile(parent.program)
#         save(params, f"session.{self.uuid}.evaparams")
#         save(signature, f"session.{self.uuid}.evasignature")

#         self.compiled = compiled_prog
#         self.session_files = [self.public_ctx_path, f"session.{self.uuid}.evaparams", f"session.{self.uuid}.evasignature"]

#     def load_public_ctx(self):
#         self.public_ctx = load(self.public_ctx_path)
#         self.state = 1

#     def execute(self, input_path):
#         self.session_files.append(input_path)
#         print("Loading input")
#         input = load(input_path)
#         print("Calculating output")
#         output = self.public_ctx.execute(self.compiled, input)
#         print("Saving output")
#         save(output, f"{input_path}.output")
#         self.session_files.append(f"{input_path}.output")
    
#     def close(self):
#         for file in self.session_files:
#             if os.path.exists(file):
#                 os.remove(file)

#         self.state = 2

# api = Flask(__name__)
# eva = EvaApp()

# # Global endpoints to retrieve vocabulary, signature and parameters

# @api.route('/vocabulary', methods=['GET'])
# def get_vocabulary():
#     if eva.vectorizer == None:
#         return {"error": "Application is not ready to accept sessions yet. Please wait..."}, 418
#     else:
#         return str(eva.vectorizer.vocabulary_).replace("\'", "\""), 200

# @api.route('/categories', methods=['GET'])
# def get_categories():
#     return {"categories": list(CATEGORIES)}, 200


# # Session endpoints to handle session management for clients

# @api.route('/session', methods=['POST'])
# def create_session():
#     id = eva.new_session()

#     if id == "":
#         return {"error": "Application is not ready to accept sessions yet. Please wait...", "session_uuid":""}, 418
#     else:
#         return {"session_uuid": id}, 200

# @api.route('/session/<id>/public', methods=['POST'])
# def set_session_context(id):
#     if id not in eva.sessions:
#         return {"error": "The requested session does not exist.", "session_state":"-1"}, 404

#     if 'file' not in request.files:
#         return {"error": "No context file provided."}, 406

#     session = eva.sessions[id]
#     file = request.files['file']
#     file.save(session.public_ctx_path)
#     session.load_public_ctx()
#     return {"session_state": "1"}, 200

# @api.route('/session/<id>/evaluate', methods=['POST'])
# def evaluate_session_input(id):
#     if id not in eva.sessions:
#         return {"error": "The requested session does not exist.", "session_state":"-1"}, 404

#     if 'file' not in request.files:
#         return {"error": "No input file provided."}, 406

#     input_uuid = uuid.uuid4().hex
#     input_fname = f"session.{id}.{input_uuid}"
#     output_fname = f"session.{id}.{input_uuid}.output"

#     print(f"Evaluating input for session {id}.")
#     session = eva.sessions[id]
#     file = request.files['file']
#     file.save(input_fname)
#     session.execute(input_fname)

#     return_data = io.BytesIO()
#     with open(output_fname, 'rb') as fo:
#         return_data.write(fo.read())
#     return_data.seek(0)

#     os.remove(input_fname)
#     os.remove(output_fname)

#     return send_file(return_data, mimetype='application/octet-stream')

# @api.route('/session/<id>/state', methods=['GET'])
# def get_session_state(id):
#     if id not in eva.sessions:
#         return {"error": "The requested session does not exist.", "session_state":"-1"}, 404
#     else:
#         return {"session_state": eva.sessions[id].state}, 200

# @api.route('/session/<id>/close', methods=['POST'])
# def close_session(id):
#     if id not in eva.sessions:
#         return {"error": "The requested session does not exist.", "session_state":"-1"}, 404
#     else:
#         eva.sessions[id].close()
#         return {"session_state": eva.sessions[id].state}, 200

# @api.route('/session/<id>/signature', methods=['GET'])
# def get_signature(id):
#     return send_file(f"session.{id}.evasignature", as_attachment=True)

# @api.route('/session/<id>/params', methods=['GET'])
# def get_params(id):
#     return send_file(f"session.{id}.evaparams", as_attachment=True)


# def run_client_process_local(idx): 
#      # Initialize Vectorizer
#     vectorizer = CountVectorizer(ngram_range=(1,3))
#     vectorizer.vocabulary_ = eva.vectorizer.vocabulary_

#     # Create server session
#     id = eva.new_session()
#     session = eva.sessions[id]

#     # Generate public-private key pair
#     params = load(f"session.{id}.evaparams")
#     public_ctx, private_ctx = generate_keys(params)
#     save(public_ctx, session.public_ctx_path)
#     session.load_public_ctx()

#     # Generate inputs
#     test = datasets.fetch_20newsgroups(
#         subset='test',
#         remove=('headers', 'footers', 'quotes'),
#         categories=CATEGORIES,
#         shuffle=False
#     )
    
#     inputs = numpy.squeeze(numpy.asarray(vectorizer.transform(test.data).todense())).tolist()

#     # Encode an input
#     signature = load(f"session.{id}.evasignature")
#     encrypted_input = public_ctx.encrypt({'x' : inputs[idx]}, signature)
#     save(encrypted_input, "client.svc.encryptedin")

#     # Send input to server
#     input_fname = "client.svc.encryptedin"
#     session.execute(input_fname)

#     # Decrypt output
#     encrypted_output = load("client.svc.encryptedin.output")
#     output = private_ctx.decrypt(encrypted_output, signature)

#     # Evaluate output
#     eval = []
#     for key in sorted(output.keys()):
#         eval.append(output[key][0])
#     i = numpy.argmax(eval)
#     print(list(CATEGORIES)[i])


#     # Encode an input
#     signature = load(f"session.{id}.evasignature")
#     encrypted_input = public_ctx.encrypt({'x' : inputs[idx + 1]}, signature)
#     save(encrypted_input, "client.svc.encryptedin")

#     # Send input to server
#     input_fname = "client.svc.encryptedin"
#     session.execute(input_fname)

#     # Decrypt output
#     encrypted_output = load("client.svc.encryptedin.output")
#     output = private_ctx.decrypt(encrypted_output, signature)

#     # Evaluate output
#     eval = []
#     for key in sorted(output.keys()):
#         eval.append(output[key][0])
#     i = numpy.argmax(eval)
#     print(list(CATEGORIES)[i])


# if __name__ == "__main__":
#     api.run()