import json
import requests
from sklearn.feature_extraction.text import CountVectorizer
from eva import load, save
from eva.seal import generate_keys
import numpy
from sklearn import datasets

ADDRESS="0.0.0.0:80"

if __name__ == "__main__":
    # Initialize Vectorizer
    print("Retrieving vocabulary from server and initializing vectorizer...")
    vectorizer = CountVectorizer(ngram_range=(1,3))
    vocab = requests.get(f"http://{ADDRESS}/vocabulary")
    json_vocab = json.loads(vocab.text.replace("'", "\""))
    vectorizer.vocabulary_ = json_vocab

    # Create server session
    print("Creating server session...")
    response = requests.request("POST", f"http://{ADDRESS}/session")
    session = response.json()['session_uuid']
    print(f"Session created with UUID {session}...")

    # Get EVA signature and parameters
    print("Retrieving encryption signature and parameters from server...")
    r = requests.get(f"http://{ADDRESS}/session/{session}/signature")  
    with open('client.svc.evasignature', 'wb') as f:
        f.write(r.content)

    r = requests.get(f"http://{ADDRESS}/session/{session}/params")  
    with open('client.svc.evaparams', 'wb') as f:
        f.write(r.content)

    # Generate public-private key pair
    print("Generating encryption context...")
    params = load('client.svc.evaparams')
    public_ctx, private_ctx = generate_keys(params)
    save(public_ctx, "client.svc.sealpublic")
    
    # Send public key to server
    print("Uploading public key to server...")
    url = f"http://{ADDRESS}/session/{session}/public"
    files=[('file',('client.svc.sealpublic',open('client.svc.sealpublic','rb'),'application/octet-stream'))]
    response = requests.request("POST", url, files=files)

    # Get categories
    categories = requests.get(f"http://{ADDRESS}/categories").json()["categories"]

    # Generate inputs
    print("Encrypting input sample...")
    test = datasets.fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        categories=tuple(categories),
        shuffle=False
    )
    
    inputs = numpy.squeeze(numpy.asarray(vectorizer.transform(test.data).todense())).tolist()

    # Encode an input
    signature = load("client.svc.evasignature")
    encrypted_input = public_ctx.encrypt({'x' : inputs[2]}, signature)
    save(encrypted_input, "client.svc.encryptedin")

    # Send input to server
    print("Sending input to server for evaluation...")
    url = f"http://{ADDRESS}/session/{session}/evaluate"
    files=[('file',('client.svc.encryptedin',open('client.svc.encryptedin','rb'),'application/octet-stream'))]
    response = requests.request("POST", url, files=files)
    with open('client.svc.encryptedout', 'wb') as f:
        f.write(response.content)

    # Decrypt output
    print("Decrypting server response...")
    encrypted_output = load("client.svc.encryptedout")
    output = private_ctx.decrypt(encrypted_output, signature)

    # Evaluate output
    print("Evaluating output classification...")
    eval = []
    for key in sorted(output.keys()):
        eval.append(output[key][0])
    i = numpy.argmax(eval)
    print(categories[i])


    # Close session
    print("Closing server session...")
    requests.post(f"http://{ADDRESS}/session/{session}/close")