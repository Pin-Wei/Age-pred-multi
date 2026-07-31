#!/usr/bin/env bash
#
# Finetunes / generates predictions from pyment multi-task model via the docker container(s). 
# source: https://github.com/estenhl/pyment-public
#
# If FastSurfer preprocessing is needed, you would have to rebuild the pyment's container
# with *.Dockerfile under folder ../etc/ 
# see how: https://hackmd.io/@Hualiteq/HJ0X91WYO#more-Images

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$SCRIPT_DIR/.."
VER=${1:-}

case $VER in
	1) 
		OUT_DIR=$ROOT_DIR/data/pyment
		if [ ! -d $OUT_DIR ]; then mkdir -p $OUT_DIR; fi
		
		docker run -it --rm \
			-v "$ROOT_DIR/data/pyment/raw":/input \
			-v "$ROOT_DIR/data/meta/pyment_labels.csv":/labels.csv \
			-v "$ROOT_DIR/etc/pyment_config.json":/configuration.json \
			-v "$ROOT_DIR/etc/licenses":/licenses \
			-v "$OUT_DIR":/output \
			--gpus all \
			--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
			estenhl/pyment-preprocess-and-finetune:latest
		;;
		
	2)
		OUT_DIR=$ROOT_DIR/data/pyment/predictions
		if [ ! -d $OUT_DIR ]; then mkdir -p $OUT_DIR; fi
		
		docker run -it --rm \
			-v "$ROOT_DIR/data/pyment/fastsurfer":/fastsurfer \
			-v "$OUT_DIR":/output \
			--gpus all \
			estenhl/pyment-predict:latest
		;;
		
	*)
		echo "Error: invalid argument."
		echo "Please specify the version of execution (1: finetune, 2: predict)."
		exit 1
		;;
esac

# Useful:
# https://github.com/estenhl/pyment-public/blob/main/docker/preprocess.Dockerfile
# https://github.com/estenhl/pyment-public/blob/main/docker/predict.Dockerfile
# https://github.com/estenhl/pyment-public/blob/main/scripts/download_weights.py
# https://github.com/estenhl/pyment-public/blob/main/pyment/cli/finetune_from_configuration.py
# https://github.com/estenhl/pyment-public/blob/main/pyment/cli/predict.py