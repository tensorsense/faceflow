FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /home

RUN apt update && apt install -y git

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/bash"]
