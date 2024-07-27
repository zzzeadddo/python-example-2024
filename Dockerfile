FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install -y wget

## Install Python dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r requirements.txt

## Download and extract model from GitHub Releases
RUN mkdir -p /challenge/model && \
    wget -O /challenge/models/best.pt "https://github.com/zzzeadddo/TEST20240727/releases/download/v1.0/best.pt" && \
    wget -O /challenge/models/best_model_II.h5 "https://github.com/zzzeadddo/TEST20240727/releases/download/v1.0/best_model_II.h5" && \
    wget -O /challenge/models/dibco_dplinknet34.th "https://github.com/zzzeadddo/TEST20240727/releases/download/v1.0/dibco_dplinknet34.th"
