IMAGE_NAME="brio_frontend"
VERSION=1.0
CONTAINER_NAME="brio"

.PHONY: help build test shell stop

help:
	@echo "- make build                 Build docker image"
	@echo "- make frontend              Start the frontend application"
	@echo "- make shell		    Open a shell inside docker image"
	@echo "- make stop		    Stop the application"	


.DEFAULT_GOAL := help

build:
	@docker build --tag ${IMAGE_NAME}:latest --tag ${IMAGE_NAME}:${VERSION} .


frontend: build
	@docker run -dp 5000:5000 \
		--name ${CONTAINER_NAME} \
		${IMAGE_NAME}

shell: 
	@docker exec -it ${CONTAINER_NAME} /bin/bash
stop:
	@docker stop ${CONTAINER_NAME}
