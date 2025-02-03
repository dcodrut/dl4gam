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

### Set up the python environment using `mamba` (see https://github.com/conda-forge/miniforge)
```shell
    mamba env create -f environment.yml
    mamba activate dl4gam
    cd dl4gam
```

### Reproduce the U-Net results
#### Data downloading (the processed NetCDF rasters):
1. Download and process the glacier outlines: check the notebook `notebooks/prepare_glacier_outlines.ipynb` 
2. Download the glacier rasters: `bash ./scripts/download_data.sh`  
   The archived rasters have ~10Gb for each year (inventory one & 2023). After extracting the NetCDF rasters, we will need ~17Gb for each year.

#### Data processing (patchifying, cross-validation splits & data statistics for normalization): 
`python main_data_prep.py`  

*Note that the generated patches need around 50Gb of disk space.*

#### Model training, testing and area estimation:
1. Train ten U-Net models for each of the five cross-validation iteration: `python main_train_ensemble.py`
(by default it runs on four GPUs and trains two models in parallel on each GPU; check the script for more details)
2. Apply all the models on each glacier, both on the inventory images and the 2023 ones, stack the ensemble predictions & estimate the areas and their uncertainties: `bash scripts/infer.sh`
3. Polygonize the rasterized predictions: `python main_polygonize.py`