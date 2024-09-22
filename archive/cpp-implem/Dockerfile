#
# Container to build Linux SEAL libraries, python wrapper, and examples
#
FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -qqy update
RUN apt-get install -qqy git cmake clang
WORKDIR /root
RUN git clone https://github.com/microsoft/SEAL.git
WORKDIR /root/SEAL
RUN git checkout tags/v3.5.9
RUN cmake . -DSEAL_BUILD_EXAMPLES=ON -DSEAL_BUILD_TESTS=ON -DSEAL_BUILD_SEAL_C=ON
RUN make
RUN make install
WORKDIR /root
RUN apt-get install -qqy python3-pip
RUN pip3 install conan
RUN git clone https://github.com/oktal/pistache.git
WORKDIR /root/pistache
RUN git checkout d8ae8ddfc28e06a71a270c5efa052a74ee7f40ce
RUN git submodule update --init
RUN mkdir build
WORKDIR /root/pistache/build
RUN cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
RUN make
RUN make install
WORKDIR /root
RUN conan profile new default --detect
RUN conan profile update settings.compiler.libcxx=libstdc++11 default
RUN pip3 install flask sklearn