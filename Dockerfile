FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update && apt-get install -y \
  gdb  \
  gdbserver \
  python3.8-dbg \
  libgl1-mesa-glx

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install https://github.com/claytantor/adeptRL/archive/main.zip

COPY sc2 /usr/src/app/sc2
COPY *.py ./
COPY .env_docker ./.env