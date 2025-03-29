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

if [ "$MODEL_NAME" == "unet" ]; then
  SEEDS=($(seq 1 10))
else
  SEEDS=("0")  # for the band ratio models (no randomness)
fi
echo "SEEDS = ${SEEDS[@]}"

if [ "$DATASET_NAME" == "s2_alps_plus" ]; then
  SPLITS=("1" "2" "3" "4" "5")
elif [ "$DATASET_NAME" == "s2_sgi" ]; then
  SPLITS=("2" "3" "4" "5")
fi
echo "SPLITS = ${SPLITS[@]}"


for SPLIT in "${SPLITS[@]}"; do
  # first make the inferences using the ensemble members, then aggregate the results and evaluate them

  for SEED in "${SEEDS[@]}"; do
    # SEED is 0 for the band ratio models (run main_band_ratio.py beforehand)
    if [ "$SEED" != "0" ]; then
      # glacier-wise inference
      echo "Inferring on DATASET_NAME = $DATASET_NAME; EVAL_SUBDIR = $EVAL_SUBDIR; VERSION = $VERSION; SPLIT = $SPLIT; SEED = $SEED"
      python main_test.py \
        --checkpoint_dir="$MODEL_ROOT_DIR/split_$SPLIT/seed_$SEED/version_$VERSION/checkpoints" \
        --fold="s_test" \
        --test_per_glacier true \
        --gpu_id=0 \
        --split_fp="../data/external/wd/$DATASET_NAME/cv_split_outlines/map_all_splits_all_folds.csv" \
        --rasters_dir="../data/external/wd/$DATASET_NAME/$EVAL_SUBDIR/glacier_wide"
    fi

    # glacier-wise evaluation
    echo "Glacier-wise evaluation for DATASET_NAME = $DATASET_NAME; EVAL_SUBDIR = $EVAL_SUBDIR; VERSION = $VERSION; SPLIT = $SPLIT; SEED = $SEED"
    python main_eval.py \
      --inference_dir="$MODEL_ROOT_DIR/split_$SPLIT/seed_$SEED/version_$VERSION/output/preds/$DATASET_NAME/$EVAL_SUBDIR" \
      --fold="s_test" \
      --rasters_dir="../data/external/wd/$DATASET_NAME/$EVAL_SUBDIR/glacier_wide"

  done

  if [ "${#SEEDS[@]}" -gt 1 ]; then
    echo "Aggregating the ensemble predictions on DATASET_NAME = $DATASET_NAME; EVAL_SUBDIR = $EVAL_SUBDIR; VERSION = $VERSION; SPLIT = $SPLIT; SEEDS = ${SEEDS[@]}"
    python main_agg_ensemble.py \
      --model_root_dir=$MODEL_ROOT_DIR \
      --split=split_$SPLIT \
      --version="version_$VERSION/" \
      --dataset_name=$DATASET_NAME \
      --eval_subdir=$EVAL_SUBDIR \
      --fold="s_test" \
      --seed_list "${SEEDS[@]}"

    # glacier-wise evaluation of the aggregated results
    echo "Evaluating the aggregated ensemble predictions on DATASET_NAME = $DATASET_NAME; EVAL_SUBDIR = $EVAL_SUBDIR; VERSION = $VERSION; SPLIT = $SPLIT; SEED = all"
    python main_eval.py \
      --inference_dir="$MODEL_ROOT_DIR/split_$SPLIT/seed_all/version_$VERSION/output/preds/$DATASET_NAME/$EVAL_SUBDIR" \
      --fold="s_test" \
      --rasters_dir="../data/external/wd/$DATASET_NAME/$EVAL_SUBDIR/glacier_wide"
  fi

done
