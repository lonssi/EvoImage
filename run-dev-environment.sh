#!/bin/bash
set -x
set -e

docker build -t evoimage .

# Cleanup old compose (if exists)
docker-compose down || true

# For display forwarding
xhost +

# Run dev environment
docker-compose run dev
