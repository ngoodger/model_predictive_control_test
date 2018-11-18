export VERSION := $(shell git rev-parse HEAD)
export GKE_PROJECT := $(shell cat GKE_PROJECT)
export MPC_IMAGE := gcr.io\/$(GKE_PROJECT)\/mpc_test:$(VERSION)
export KUBE_YAML_IMAGE_INDENT := \ \ \ \ \ \ \ \ \ \ 

install:
	cp scripts/pre-commit .git/hooks/

clean:
	rm frames/*

format:
	ABSOLUTE_PROJECT_SRC_DIR="${PWD}/src/main/python ${PWD}/src/test/python" \
    source scripts/define_python_format_function.sh && \
    format_python

format_check:
	ABSOLUTE_PROJECT_SRC_DIR="${PWD}/src/main/python ${PWD}/src/test/python" \
    source scripts/define_python_format_function.sh && \
    format_python_check

train_model:
	python3 src/main/python/train_model.py

train_policy:
	python3 src/main/python/train_policy.py

run_model_show_frames:
	python3 src/main/python/run_model_show_frames.py

run_policy_show_frames:
	python3 src/main/python/run_policy_show_frames.py

build_image:
	docker build . -t mpc_test:$(VERSION)

update_deployment_version:
	cat kube_train_model.yaml | sed 's/^.*- image:.*$$/$(KUBE_YAML_IMAGE_INDENT)- image: $(MPC_IMAGE)/' > kube_train_model.yaml.bak
	mv kube_train_model.yaml.bak kube_train_model.yaml

push_image_gke: build_image
	docker tag mpc_test:$(VERSION) gcr.io/$(GKE_PROJECT)/mpc_test:$(VERSION)
	docker push gcr.io/$(GKE_PROJECT)/mpc_test:$(VERSION)
