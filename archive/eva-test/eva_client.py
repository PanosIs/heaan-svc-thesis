import json
import requests
from sklearn.feature_extraction.text import CountVectorizer
from eva import load, save
from eva.seal import generate_keys
import numpy
from sklearn import datasets

if __name__ == "__main__":
    response = requests.request("POST", 'http://127.0.0.1:5000/localrun')
    # #####################################################
    # r = requests.get(f"http://127.0.0.1:5000/params")  
    # with open('client.evaparams', 'wb') as f:
    #     f.write(r.content)

    # print('Getting parameters from server')
    # r = requests.get(f"http://127.0.0.1:5000/signature")  
    # with open('client.evasignature', 'wb') as f:
    #     f.write(r.content)
    # #####################################################
    # print('Key generation time')
    # params = load('client.evaparams')

    # public_ctx, secret_ctx = generate_keys(params)

    # save(public_ctx, 'client.sealpublic')
    # save(secret_ctx, 'client.sealsecret')
    
    # #####################################################
    # url = f"http://127.0.0.1:5000/public"
    # files=[('file',('client.sealpublic',open('client.sealpublic','rb'),'application/octet-stream'))]
    # response = requests.request("POST", url, files=files)
    # #####################################################
    # print('Runtime on client')
    # signature = load('client.evasignature')
    # public_ctx = load('client.sealpublic')

    # inputs = {
    #     'x': [i for i in range(signature.vec_size)]
    # }
    # encInputs = public_ctx.encrypt(inputs, signature)

    # save(encInputs, 'client.input.sealvals')
    # #####################################################
    # url = f"http://127.0.0.1:5000/evaluate"
    # files=[('file',('client.input.sealvals',open('client.input.sealvals','rb'),'application/octet-stream'))]
    # response = requests.request("POST", url, files=files)
    # with open('client.output.sealvals', 'wb') as f:
    #     f.write(response.content)
    # #####################################################
    # print('Back on client')
    # secret_ctx = load('poly.sealsecret')
    # encOutputs = load('poly_outputs.sealvals')

    # outputs = secret_ctx.decrypt(encOutputs, signature)

    # reference = evaluate(poly, inputs)
    # print('Expected', reference)
    # print('Got', outputs)
    # print('MSE', valuation_mse(outputs, reference))