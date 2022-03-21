import json
import requests
from sklearn.feature_extraction.text import CountVectorizer
from eva import load, save
from eva.seal import generate_keys
import numpy
from sklearn import datasets

if __name__ == "__main__":
    # Initialize Vectorizer
    vectorizer = CountVectorizer(ngram_range=(1,3))
    vocab = requests.get("http://127.0.0.1:5000/vocabulary")
    json_vocab = json.loads(vocab.text.replace("'", "\""))
    vectorizer.vocabulary_ = json_vocab

    # Create server session
    response = requests.request("POST", "http://127.0.0.1:5000/session")
    session = response.json()['session_uuid']

    # Get EVA signature and parameters
    r = requests.get(f"http://127.0.0.1:5000/session/{session}/signature")  
    with open('client.svc.evasignature', 'wb') as f:
        f.write(r.content)

    r = requests.get(f"http://127.0.0.1:5000/session/{session}/params")  
    with open('client.svc.evaparams', 'wb') as f:
        f.write(r.content)

    # Generate public-private key pair
    params = load('client.svc.evaparams')
    public_ctx, private_ctx = generate_keys(params)
    save(public_ctx, "client.svc.sealpublic")

    
    # Send public key to server
    url = f"http://127.0.0.1:5000/session/{session}/public"
    files=[('file',('client.svc.sealpublic',open('client.svc.sealpublic','rb'),'application/octet-stream'))]
    response = requests.request("POST", url, files=files)

    # Get categories
    categories = requests.get("http://127.0.0.1:5000/categories").json()["categories"]

    # Generate inputs
    test = datasets.fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        categories=tuple(categories),
        shuffle=False
    )
    
    inputs = numpy.squeeze(numpy.asarray(vectorizer.transform(test.data).todense())).tolist()

    # Encode an input
    signature = load("client.svc.evasignature")
    encrypted_input = public_ctx.encrypt({'x' : inputs[1]}, signature)
    save(encrypted_input, "client.svc.encryptedin")

    # Send input to server
    url = f"http://127.0.0.1:5000/session/{session}/evaluate"
    files=[('file',('client.svc.encryptedin',open('client.svc.encryptedin','rb'),'application/octet-stream'))]
    response = requests.request("POST", url, files=files)
    with open('client.svc.encryptedout', 'wb') as f:
        f.write(response.content)

    # Decrypt output
    encrypted_output = load("client.svc.encryptedout")
    output = private_ctx.decrypt(encrypted_output, signature)

    # Evaluate output
    eval = []
    for key in sorted(output.keys()):
        eval.append(output[key][0])
    i = numpy.argmax(eval)
    print(categories[i])


    # Close session
    requests.post(f"http://127.0.0.1:5000/session/{session}/close")