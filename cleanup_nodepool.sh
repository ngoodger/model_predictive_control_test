#!/bin/bash
CLUSTER_NAME=$(cat GKE_CLUSTER_NAME)
gcloud container node-pools delete gpu-pool --cluster=$CLUSTER_NAME --zone=us-central1-a --quiet
gcloud container node-pools delete cpu-pool --cluster=$CLUSTER_NAME --zone=us-central1-a --quiet

