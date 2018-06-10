#!/bin/bash
set -x
set -e

docker build -t evoimage .

# Cleanup old compose (if exists)
docker-compose down || true

# For display forwarding
if [ "$(uname)" == "Darwin" ]
    ip=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
    xhost + $ip
    DISPLAY=$ip:0
then
   echo MacOS
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]
then
    echo Linux
    xhost +
elif [ -n "$COMSPEC" -a -x "$COMSPEC" ]
then
    echo $0: this script does not support Windows \:\(
    exit
fi

# Run dev environment
docker-compose run dev
