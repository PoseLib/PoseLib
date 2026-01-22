#FROM ubuntu:latest AS poselib
FROM ubuntu:22.04 AS poselib

RUN apt-get update && \
    apt-get install -y libeigen3-dev cmake build-essential

ADD . /PoseLib
WORKDIR /PoseLib

#
# How to build?
#
#   docker build -t poselib -f Dockerfile .
#
# How to run?
#
#   > docker run -it poselib bash
#
#   (inside docker image -> compile project)
#   > scripts/build_linux.sh
#
