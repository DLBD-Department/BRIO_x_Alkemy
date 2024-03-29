IMAGE_NAME="brio_frontend"
CONTAINER_NAME="brio"

UNAME := $(shell uname -o)
ifeq ($(UNAME), GNU/Linux)
    HOST_IP=$(shell hostname -I | cut -d ' ' -f1)
endif
ifeq ($(UNAME), Darwin)
    HOST_IP=$(shell osascript -e "IPv4 address of (system info)")
endif

.PHONY: help build test shell stop

help:
	@echo "- make build         Build docker image"
	@echo "- make frontend      Start the frontend application is a container"
	@echo "- make shell		    Open a shell inside docker container"
	@echo "- make stop		    Stop the docker container"


.DEFAULT_GOAL := help

.PHONY: build
build:
	@docker build \
		--tag ${IMAGE_NAME}:latest \
		--tag ${IMAGE_NAME}:$$(cat VERSION.txt) \
		.

.PHONY: frontend
frontend: build
	@docker run -dp 5000:5000 \
		--name ${CONTAINER_NAME} \
		--env HOST_IP=$(HOST_IP) \
		${IMAGE_NAME}

.PHONY: shell
shell:
	@docker exec -it ${CONTAINER_NAME} /bin/bash

.PHONY: stop
stop:
	@docker stop ${CONTAINER_NAME}
	@docker rm ${CONTAINER_NAME}

.PHONY: test
test:
	@python3 -m tests.unit.TestBiasDetector

.PHONY: venv
venv:
	python3 -m virtualenv .
	. bin/activate; pip install -Ur requirements.txt
	. bin/activate; pip install -Ur requirements-dev.txt

.PHONY: clean
clean:
	-rm -rf build dist
	-rm -rf *.egg-info
	-rm -rf bin lib share pyvenv.cfg
