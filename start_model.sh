#!/bin/bash
export WORLD_SIZE=1
export MASTER_PORT=8080
export MASTER_ADDR=127.0.0.1
export RANK=0
export GCS_BUCKET=mpc-test
export GOOGLE_APPLICATION_CREDENTIALS=gs_key.json
python3 src/main/python/train_model.py
#export RANK=1
#python3 src/main/python/train_model.py &
#export RANK=2
#python3 src/main/python/train_model.py &
#export RANK=3
#python3 src/main/python/train_model.py &
