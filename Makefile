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
	docker build . -t mpc_test

push_image_gke: build_image
	docker tag mpc_test gcr.io/t-pulsar-217904/mpc_test
	docker push gcr.io/t-pulsar-217904/mpc_test
