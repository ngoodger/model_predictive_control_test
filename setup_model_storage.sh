#!/bin/bash
BUCKET_NAME=mpc-test
PROJECT_ID=proud-spring-222310
gcloud iam service-accounts create kubeflow-codelab --display-name kubeflow-codelab
IAM_EMAIL=kubeflow-codelab@$PROJECT_ID.iam.gserviceaccount.com
gsutil acl ch -u $IAM_EMAIL:O gs://$BUCKET_NAME
gcloud iam service-accounts keys create ./gs_key.json \
      --iam-account=$IAM_EMAIL
