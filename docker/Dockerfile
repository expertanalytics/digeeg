FROM ubuntu:19.10

USER root

ENV HOME /usr/home

RUN mkdir -p /usr/home/shared
WORKDIR $HOME
VOLUME $HOME/shared

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install python3-pip git libsm6 libxrender1 && \
    apt-get -y install vim && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install scipy opencv-python numpy shapely matplotlib imutils pylint h5py

RUN cd $HOME && git clone -b js git://github.com/expertanalytics/digeeg && cd digeeg && \
    python3 -m pip install -e .
