#!/bin/bash
set -e
cd $(git rev-parse --show-toplevel)

echo PORT is set to 8000
docker run --rm -ti -p 8000:80 smart_image_resizer
