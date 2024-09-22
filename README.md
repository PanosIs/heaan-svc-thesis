# Homomorphic Encryption for Machine Learning with Support Vector Machines

Homomorphic encryption (HE) has emerged as a cryptographic technique that enables computations to be performed on encrypted data without decrypting it. This capability holds promise for preserving data privacy and security in various domains, particularly in the realm of artificial intelligence (AI) and machine learning (ML) which are particularly relevant at present. By leveraging HE, sensitive ML models and their associated data can be securely outsourced to cloud services or shared with collaborators, without compromising privacy or intellectual property.

This thesis investigates the potential applications of homomorphic encryption in the context of machine learning. We explore how HE can be employed to protect the privacy of both machine learning models as well as client data and samples.

## How to run the demo

Create a network for the containers to connect to each other locally:
```
docker network create heaan-svc-network
```

Deploy the server locally: 
```
cd demo/server
docker build . --tag heaan-svc-server 
docker run --name heaan-svc-server --network heaan-svc-network -d -p 80:80 heaan-svc-server
```

Run the client locally in docker in interactive mode: 
```
cd ../client
docker build . --tag heaan-svc-client
EVA_SERVER_ADDRESS=`docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' heaan-svc-server` && export EVA_SERVER_ADDRESS
docker run -e EVA_SERVER_ADDRESS=$EVA_SERVER_ADDRESS --network heaan-svc-network -it heaan-svc-client
```

## How to run the demo client with a remote server

Given a remote server address where the code runs you should be able to run the demo client only and connect to the remote server:
```
docker run -e EVA_SERVER_ADDRESS=<REMOTE_SERVER_ADDRESS> -it ghcr.io/dreammify/heaan-svc-thesis:client                                                                                 
```

Depending on your machine, this may produce a core dump if the correct image version hasn't been built remotely (images are available for `linux/amd64` and `linux/arm64/v8`, usually docker will show a warning if an image is not available for your platform) This is because the C++ binaries are built on the image and are not multiplatform. If this happens, simply build the image locally as above:
```
cd demo/client
docker build . --tag heaan-svc-client
docker run -e EVA_SERVER_ADDRESS=<REMOTE_SERVER_ADDRESS> -it heaan-svc-client
```