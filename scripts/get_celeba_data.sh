#!/bin/bash
# Download kaggle celeba dataset
# Note this requires an API key downloaded from a kaggle account, at ~/.kaggle/kaggle.json 

set -e

PROJ_ROOT=$(dirname $(dirname $(realpath $0)))
DOWNLOAD_DIR="$PROJ_ROOT/data/celeba_dataset"

mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

kaggle datasets download -d jessicali9530/celeba-dataset --unzip

printf "Downloaded celeba dataset to $DOWNLOAD_DIR \n"
