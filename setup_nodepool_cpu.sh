#!/bin/bash
CLUSTER_NAME=$(cat GKE_CLUSTER_NAME)
gcloud container node-pools create cpu-pool --cluster=$CLUSTER_NAME --machine-type=n1-highcpu-4 --zone=us-central1-a --num-nodes=1 --preemptible
