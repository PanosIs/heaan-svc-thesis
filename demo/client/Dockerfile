# Container to build Linux SEAL libraries, python wrapper, and examples
FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qqy 
RUN apt-get install -qqy git cmake libboost-all-dev libprotobuf-dev protobuf-compiler
# Install clang as a C compiler
RUN apt-get install -qqy clang
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100
RUN update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100
# Install Python dependencies
RUN apt-get install -qqy pip
RUN python3 -m pip install numpy flask requests scikit-learn click
# Install Microsoft SEAL
WORKDIR /root
RUN git clone -b v3.6.4 https://github.com/microsoft/SEAL.git
WORKDIR /root/SEAL
RUN cmake -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF .
RUN make -j
RUN make install
# Build EVA
WORKDIR /root
RUN git clone -b v1.0.1 https://github.com/microsoft/EVA.git
WORKDIR /root/EVA
RUN git submodule update --init
RUN cmake -DUSE_GALOIS=ON .
RUN make -j
RUN python3 -m pip install -e python/
# Run tests
RUN python3 tests/all.py
# Copy server file
COPY . .
ENTRYPOINT /usr/bin/python3 eva_client.py