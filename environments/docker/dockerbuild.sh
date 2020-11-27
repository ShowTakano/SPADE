#!/bin/sh
docker build -t pytorch/spade:201120 .
# $ docker run --runtime=nvidia -it --rm -v /work1:/work1 -e NVIDIA_VISIBLE_DEVICES=0 pytorch/spade:201120 /bin/bash 