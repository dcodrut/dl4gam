## DL4GAM: a multi-modal Deep Learning-based framework for Glacier Area Monitoring, trained and validated on the European Alps

DL4GAM was developed to monitor glacier area changes using Deep Learning models trained on multi-modal data (so far, optical, DEM, and elevation change data).

The framework is glacier-focused, meaning that the data is processed and analyzed on a glacier-by-glacier basis. This allows us to download the best images independently for each glacier, minimizing as much as possible both the cloud/shadow coverage and the seasonal snow. The glacier-focus approach should also make it easier to adapt the framework to other regions/setups.

Currently, the models ([U-Net](https://arxiv.org/abs/1505.04597) ensemble) are trained and validated on the European Alps, in a 5-fold cross-validation setup, so we can use only the testing folds when later computing the area changes. 

Other resources:
- preprint: https://doi.org/10.22541/essoar.173557607.70204641/v1
- predicted glacier outlines: https://dcodrut.github.io/dl4gam_alps/
- dataset description: https://huggingface.co/datasets/dcodrut/dl4gam_alps
- model checkpoints: https://huggingface.co/dcodrut/dl4gam_alps

For reproducing the results, please follow the steps below. Please open an issue if you encounter any problems.

### Set up the python environment using [mamba](https://github.com/conda-forge/miniforge)
```shell
mamba env create -f environment.yml
mamba activate dl4gam
cd dl4gam
```

### Reproduce the results for the European Alps
#### 1. Data downloading (the inventory outlines and the processed NetCDF rasters):
- Download and process the glacier outlines: check `notebooks/prepare_glacier_outlines.ipynb`
- Download the glacier rasters (Note that the archives have ~10Gb for each year, i.e. the inventory one & 2023; after extracting the NetCDF rasters, we will need ~17Gb for each year):   
    ```shell
    bash ./scripts/s2_alps_plus/download_data.sh
    ```
#### 2. Data processing for training (patchifying, cross-validation splits & data statistics for normalization):
```shell
python main_prep_data_train.py
```

Note that the generated patches need around 50Gb of disk space.

#### 3. Model training & inference:
- Train ten U-Net models for each of the five cross-validation iteration (adjust the parameters depending on the available hardware):   
```shell
python main_train_ensemble.py \
    --config_fp "./configs/s2_alps_plus/unet.yaml" \
    --n_splits 5 \
    --ensemble_size 10 \
    --n_gpus 4 \
    --max_tasks_per_gpu 2 
```
- Apply the models on each glacier from the test folds, both on the inventory and the 2023 data, stack the ensemble predictions & estimate the areas: 
```shell
export S2_ALPS_YEAR=inv; bash scripts/s2_alps_plus/infer_and_eval.sh
export S2_ALPS_YEAR=2023; bash scripts/s2_alps_plus/infer_and_eval.sh
```

#### 4. Post-processing:
- Compute the area changes and their uncertainty using the ensemble-averaged predictions:   
```shell
python main_postprocess_stats.py \
  --model_dir "../data/external/_experiments/s2_alps_plus/unet" \
  --model_version "version_0" \
  --seed_list "all" \
  --split_list 1 2 3 4 5 \
  --subdir_list "inv" "2023" 
```
- Polygonize the ensemble-averaged predictions
```shell
python main_polygonize.py \
  --model_dir "../data/external/_experiments/s2_alps_plus/unet" \
  --model_version "version_0" \
  --split_list 1 2 3 4 5 \
  --seed "all"
```