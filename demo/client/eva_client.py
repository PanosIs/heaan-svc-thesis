import json
import requests
import click
import random
import os
from sklearn.feature_extraction.text import CountVectorizer
from eva import load, save
from eva.seal import generate_keys
import numpy
from sklearn import datasets

class Client:
    def __init__(self) -> None:
        # Initialize Vectorizer
        print("Retrieving vocabulary from server and initializing vectorizer...")
        self.vectorizer = CountVectorizer(ngram_range=(1,3))
        vocab = requests.get(f"http://{ADDRESS}/vocabulary")
        json_vocab = json.loads(vocab.text.replace("'", "\""))
        self.vectorizer.vocabulary_ = json_vocab

        # Create server session
        print()
        print("Creating server session...")
        response = requests.request("POST", f"http://{ADDRESS}/session")
        self.session = response.json()['session_uuid']
        print(f"Session created with UUID {self.session}...")

        # Get EVA signature and parameters
        print("Retrieving encryption signature and parameters from server...")
        r = requests.get(f"http://{ADDRESS}/session/{self.session}/signature")  
        with open('client.svc.evasignature', 'wb') as f:
            f.write(r.content)

        r = requests.get(f"http://{ADDRESS}/session/{self.session}/params")  
        with open('client.svc.evaparams', 'wb') as f:
            f.write(r.content)

        # Generate public-private key pair
        print("Generating encryption context...")
        params = load('client.svc.evaparams')
        self.public_ctx, self.private_ctx = generate_keys(params)
        save(self.public_ctx, "client.svc.sealpublic")

        # Send public key to server
        print("Uploading public key to server...")
        url = f"http://{ADDRESS}/session/{self.session}/public"
        files=[('file',('client.svc.sealpublic',open('client.svc.sealpublic','rb'),'application/octet-stream'))]
        response = requests.request("POST", url, files=files)

        # Get categories
        self.categories = requests.get(f"http://{ADDRESS}/categories").json()["categories"]

    # If index == -1 pick random index
    def classify_indexed(self, index):
        # Generate inputs
        print("Encrypting input sample...")
        test = datasets.fetch_20newsgroups(
            subset='test',
            remove=('headers', 'footers', 'quotes'),
            categories=tuple(self.categories),
            shuffle=False
        )

        if index == -1:
            index = random.randint(0, len(test.data))

        inputs = numpy.squeeze(numpy.asarray(self.vectorizer.transform(test.data).todense())).tolist()

        print("Selected sample:")
        print()
        print(test.data[index])
        print()
    
        # Encode an input
        signature = load("client.svc.evasignature")
        encrypted_input = self.public_ctx.encrypt({'x' : inputs[index]}, signature)
        save(encrypted_input, "client.svc.encryptedin")

        # Send input to server
        print("Sending input to server for evaluation...")
        url = f"http://{ADDRESS}/session/{self.session}/evaluate"
        files=[('file',('client.svc.encryptedin',open('client.svc.encryptedin','rb'),'application/octet-stream'))]
        response = requests.request("POST", url, files=files)
        with open('client.svc.encryptedout', 'wb') as f:
            f.write(response.content)

        # Decrypt output
        print("Decrypting server response...")
        encrypted_output = load("client.svc.encryptedout")
        output = self.private_ctx.decrypt(encrypted_output, signature)

        # Evaluate output
        print("Evaluating output classification...")
        eval = []
        for key in sorted(output.keys()):
            eval.append(output[key][0])
        i = numpy.argmax(eval)
        print()
        print("Classification:")
        print(self.categories[i])
        print("Actual classification:")
        print(self.categories[test.target[index]])
        print()

    def classify_custom(self, input):
        inputs = numpy.squeeze(numpy.asarray(self.vectorizer.transform([input]).todense())).tolist()
    
        # Encode an input
        signature = load("client.svc.evasignature")
        encrypted_input = self.public_ctx.encrypt({'x' : inputs}, signature)
        save(encrypted_input, "client.svc.encryptedin")

        # Send input to server
        print("Sending input to server for evaluation...")
        url = f"http://{ADDRESS}/session/{self.session}/evaluate"
        files=[('file',('client.svc.encryptedin',open('client.svc.encryptedin','rb'),'application/octet-stream'))]
        response = requests.request("POST", url, files=files)
        with open('client.svc.encryptedout', 'wb') as f:
            f.write(response.content)

        # Decrypt output
        print("Decrypting server response...")
        encrypted_output = load("client.svc.encryptedout")
        output = self.private_ctx.decrypt(encrypted_output, signature)

        # Evaluate output
        print("Evaluating output classification...")
        eval = []
        for key in sorted(output.keys()):
            eval.append(output[key][0])
        i = numpy.argmax(eval)
        print()
        print("Classification:")
        print(self.categories[i])
        print()

    def close_session(self):
        # Close session
        print("Closing server session...")
        requests.post(f"http://{ADDRESS}/session/{self.session}/close")

ADDRESS=os.getenv("EVA_SERVER_ADDRESS", "0.0.0.0:80")

@click.command()
def classify():
    client = Client()

    mode = click.prompt(
        'Choose client. Random will pick random samples to classify, indexed will allow you to specify the samples yourself while custom will allow you to enter custom input', 
        type=click.Choice(['random', 'indexed', 'custom'], case_sensitive=False),
        default='random',
    )

    iterations = click.prompt('Choose number of iterations (How many samples to classify), default 1', type=int, default=1)

    for iteration in range(iterations):
        if mode == 'random':
            client.classify_indexed(-1)
        elif mode == 'indexed':
            index = click.prompt('Enter the index of the sample to classify', type=int)
            client.classify_indexed(index)
        elif mode == 'custom':
            input = click.prompt('Enter the input to classify', type=str)
            client.classify_custom(input)

    client.close_session()

if __name__ == '__main__':
    classify()