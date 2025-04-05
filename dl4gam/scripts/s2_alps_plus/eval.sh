#!/bin/bash

# show the working directory
echo "Working directory: $PWD"

VERSION="0"
echo "VERSION = $VERSION"

# on which dataset to make inferences
DATASET_NAME="s2_alps_plus"
#DATASET_NAME="s2_sgi"  # SGI2016 inventory

# model
MODEL_NAME="unet"
#MODEL_NAME="band_ratio_glacier-wide"
#MODEL_NAME="band_ratio_regional"

MODEL_ROOT_DIR="../data/external/_experiments/s2_alps_plus/$MODEL_NAME"

echo "MODEL_ROOT_DIR = $MODEL_ROOT_DIR"

# set the evaluation subdirectory (use the environment variable if it is set)
if [[ -z "${S2_ALPS_YEAR}" ]]; then
  EVAL_SUBDIR="inv"
else
  EVAL_SUBDIR="${S2_ALPS_YEAR}"
fi
echo "EVAL_SUBDIR = $EVAL_SUBDIR"

if [[ $MODEL_NAME == "unet" ]]; then
  SEEDS=($(seq 1 10))
else
  SEEDS=("0")  # for the band ratio models (no randomness)
fi
echo "SEEDS = ${SEEDS[@]}"

if [[ $DATASET_NAME == "s2_alps_plus" ]]; then
  SPLITS=("1" "2" "3" "4" "5")
elif [ "$DATASET_NAME" == "s2_sgi" ]; then
  SPLITS=("2" "3" "4" "5")
fi
echo "SPLITS = ${SPLITS[@]}"

if [[ $EVAL_SUBDIR == "inv" ]]; then
  # wee need to evaluate the models on the validation set first so we can calibrate them
  FOLDS=("s_valid" "s_test")
else
  FOLDS=("s_test")
fi
echo "FOLDS = ${FOLDS[@]}"

for SPLIT in "${SPLITS[@]}"; do
  for FOLD in "${FOLDS[@]}"; do
    DESC="DATASET_NAME = $DATASET_NAME; EVAL_SUBDIR = $EVAL_SUBDIR; VERSION = $VERSION; SPLIT = $SPLIT; FOLD = $FOLD"
    echo -e "\n$DESC"

    # first make the inferences using the ensemble members (optional), then aggregate the results and evaluate them
    for SEED in "${SEEDS[@]}"; do
      # glacier-wise evaluation
      echo "Glacier-wise evaluation for $DESC; SEED = $SEED"
      python main_eval.py \
        --inference_dir="$MODEL_ROOT_DIR/split_$SPLIT/seed_$SEED/version_$VERSION/output/preds/$DATASET_NAME/$EVAL_SUBDIR" \
        --fold=$FOLD \
        --rasters_dir="../data/external/wd/$DATASET_NAME/$EVAL_SUBDIR/glacier_wide"
    done

    if [ "${#SEEDS[@]}" -gt 1 ]; then
      echo "Aggregating the ensemble predictions on $DESC; SEEDS = ${SEEDS[@]}"
      python main_agg_ensemble.py \
        --model_root_dir=$MODEL_ROOT_DIR \
        --split=split_$SPLIT \
        --version="version_$VERSION/" \
        --dataset_name=$DATASET_NAME \
        --eval_subdir=$EVAL_SUBDIR \
        --fold=$FOLD \
        --seed_list "${SEEDS[@]}"

      # glacier-wise evaluation of the aggregated results (before calibration)
      echo "Evaluating the aggregated ensemble predictions (uncalibrated) on $DESC; SEED = all"
      python main_eval.py \
        --inference_dir="$MODEL_ROOT_DIR/split_$SPLIT/seed_all/version_$VERSION/output/preds/$DATASET_NAME/$EVAL_SUBDIR" \
        --fold=$FOLD \
        --rasters_dir="../data/external/wd/$DATASET_NAME/$EVAL_SUBDIR/glacier_wide" \

      # calibrate the ensemble
      # (first the individual members, at pixel level, then the aggregated results, at glacier-area level)
      python main_calib.py \
        --inference_dir="$MODEL_ROOT_DIR/split_$SPLIT/seed_all/version_$VERSION/output/preds/$DATASET_NAME/$EVAL_SUBDIR" \
        --fold=$FOLD \
        --rasters_dir="../data/external/wd/$DATASET_NAME/$EVAL_SUBDIR/glacier_wide"

      # glacier-wise evaluation of the aggregated results (after calibration, both at pixel and glacier-area level)
      echo "Evaluating the aggregated ensemble predictions (calibrated) on $DESC; SEED = all"
      python main_eval.py \
        --inference_dir="$MODEL_ROOT_DIR/split_$SPLIT/seed_all/version_$VERSION/output/preds_calib/$DATASET_NAME/$EVAL_SUBDIR" \
        --fold=$FOLD \
        --rasters_dir="../data/external/wd/$DATASET_NAME/$EVAL_SUBDIR/glacier_wide" \

    fi

  done
done
