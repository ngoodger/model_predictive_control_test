CLUSTER_NAME=standard-cluster-3
gcloud container clusters get-credentials $CLUSTER_NAME --zone us-central1-a
kubectl create clusterrolebinding default-admin \
      --clusterrole=cluster-admin --user=$(gcloud config get-value account)
