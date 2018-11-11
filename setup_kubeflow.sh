# To download ksonnet for linux (including Cloud Shell)
KS_VER=ks_0.11.0_linux_amd64

# To download ksonnet for macOS
KS_VER=ks_0.11.0_darwin_amd64

# Download tar of ksonnet
wget --no-check-certificate \
https://github.com/ksonnet/ksonnet/releases/download/v0.11.0/$KS_VER.tar.gz

# Unpack file
tar -xvf $KS_VER.tar.gz

# Add ks command to path
PATH=$PATH:$(pwd)/$KS_VER

ks init ksonnet-kubeflow
cd ksonnet-kubeflow
ks env add cloud
VERSION=v0.2.0-rc.1
ks registry add kubeflow github.com/kubeflow/kubeflow/tree/${VERSION}/kubeflow
ks pkg install kubeflow/pytorch-job
ks generate pytorch-operator pytorch-operator
ks apply cloud -c pytorch-operator
