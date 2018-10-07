FROM ubuntu:16.04
ADD . /project
WORKDIR /project
RUN apt-get update 
RUN apt-get install -y python3-pip make
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
