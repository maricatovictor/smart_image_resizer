#!/bin/bash
set -e
cd $(git rev-parse --show-toplevel)

docker build -f streamlit/Dockerfile -t smart_image_resizer .
