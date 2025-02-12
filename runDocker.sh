#!/bin/bash

docker pull woongsu0614/animal-predict:latest
docker run -p 8000:8000 -it -d --name animal-predict-container woongsu0614/animal-predict:latest
