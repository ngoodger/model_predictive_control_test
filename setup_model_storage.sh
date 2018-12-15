#!/bin/bash
BUCKET_NAME=model-predictive-control-test
PROJECT_ID=$(shell cat GKE_PROJECT)
gcloud iam service-accounts create kubeflow-codelab --display-name kubeflow-codelab
IAM_EMAIL=kubeflow-codelab@$PROJECT_ID.iam.gserviceaccount.com
gsutil acl ch -u $IAM_EMAIL:O gs://$BUCKET_NAME
gcloud iam service-accounts keys create ./gs_key.json \
      --iam-account=$IAM_EMAIL
