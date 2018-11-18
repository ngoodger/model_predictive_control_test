CLUSTER_NAME=$(cat GKE_CLUSTER_NAME)
gcloud container clusters get-credentials $CLUSTER_NAME --zone us-central1-a
kubectl create clusterrolebinding default-admin \
      --clusterrole=cluster-admin --user=$(gcloud config get-value account)
