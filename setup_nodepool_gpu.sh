#!/bin/bash
CLUSTER_NAME=$(cat GKE_CLUSTER_NAME)
gcloud container node-pools create cpu-pool --cluster=$CLUSTER_NAME --machine-type=n1-standard-2 --zone=us-central1-a --num-nodes=1 --preemptible
gcloud container node-pools create gpu-pool --cluster=$CLUSTER_NAME --machine-type=n1-standard-4 --accelerator=type=nvidia-tesla-p4,count=4 --enable-autoscaling --max-nodes=1 --min-nodes=0 --preemptible --zone=us-central1-a --num-nodes=0 --disk-size=30 --disk-type=pd-ssd
