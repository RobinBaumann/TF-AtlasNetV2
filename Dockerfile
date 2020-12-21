FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y git wget tar libgl1-mesa-glx
RUN pip3 --no-cache-dir install \
    pandas numpy open3d==0.9.0.0 mlflow absl-py

COPY . /env-atlasnet

WORKDIR /env-atlasnet