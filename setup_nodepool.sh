#!/bin/bash
CLUSTER_NAME=$(cat GKE_CLUSTER_NAME)
gcloud container node-pools create cpu-pool --cluster=$CLUSTER_NAME --machine-type=n1-highcpu-2 --zone=us-central1-a --num-nodes=1 --preemptible
gcloud container node-pools create gpu-pool --cluster=$CLUSTER_NAME --machine-type=n1-standard-2 --accelerator=type=nvidia-tesla-p4,count=1 --enable-autoscaling --max-nodes=1 --min-nodes=0 --preemptible --zone=us-central1-a --num-nodes=0
