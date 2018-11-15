FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
RUN apt-get update 
RUN apt-get install -y python3-pip make
RUN pip3 install --upgrade pip
ADD ./requirements.txt /project/requirements.txt
RUN pip3 install -r /project/requirements.txt
WORKDIR /project
ADD ./src /project/src
ADD ./Makefile /project/
