FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04
RUN apt-get update 
RUN apt-get install -y python3-pip make
RUN pip3 install --upgrade pip
ADD ./requirements.txt /project/requirements.txt
RUN pip3 install -r /project/requirements.txt
WORKDIR /project
ADD ./src /project/src
ADD ./Makefile /project/
ENTRYPOINT ["python3", "/project/src/main/python/train_model.py"]
