# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-11.html#rel_20-11
FROM nvcr.io/nvidia/pytorch:20.11-py3

# Setting Proxy
#ENV http_proxy  "http://xxx.xxx.xxx.xxx:xxxx/"
#ENV https_proxy "http://xxx.xxx.xxx.xxx:xxxx/"
#ENV ftp_proxy   "http://xxx.xxx.xxx.xxx:xxxx/"

# env
# If your docker-image baseed on ubuntu18.04, "noninteractive" is need. due to tzdata.
ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-c"]

# Setup Timezone Asia/Tokyo
RUN    apt-get update \
    && apt-get install -y tzdata \
    && echo "Asia/Tokyo" > /etc/timezone   \
    && rm  /etc/localtime        \
    && ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# apt update
#RUN sed -i -e 's@archive.ubuntu.com\|security.ubuntu.com@ftp.riken.jp/Linux@g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libfontconfig1 libxrender1 git libboost-dev libpq-dev \
  libsm6 libxext-dev g++ curl lib32stdc++6 libleveldb-dev liblmdb-dev libhdf5-dev python3 python3-dev python3-pip \
  vim zip wget cmake apt-utils sudo language-pack-en language-pack-ja software-properties-common openssh-server libgl1-mesa-dev \
  fonts-freefont-ttf \
  postgresql-client postgresql-client-common \
  && apt-get clean

# RUN add-apt-repository ppa:kelleyk/emacs && apt-get install -y --no-install-recommends emacs
RUN update-locale LANG=en_US.UTF-8 && update-locale LANG=ja_JP.UTF-8

RUN pip install torch>=1.0.0
RUN pip install torchvision
RUN pip install dominate>=2.3.1
RUN pip install dill
RUN pip install scikit-image
RUN pip install opencv-python==4.1.1.26