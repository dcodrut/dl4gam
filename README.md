## DL4GAM: a multi-modal Deep Learning-based framework for Glacier Area Monitoring, trained and validated on the European Alps

DL4GAM was developed to monitor glacier area changes using Deep Learning models trained on multi-modal data — currently including optical imagery, DEMs, and elevation change data.

The framework is glacier-focused, meaning that data is processed and analyzed on a glacier-by-glacier basis. 
This allows us to download the best images independently for each glacier, minimizing cloud/shadow coverage and seasonal snow as much as possible. 
This glacier-centric approach also makes the framework easier to adapt to other regions or setups.

Currently, the models ([U-Net](https://arxiv.org/abs/1505.04597) ensemble) are trained and validated on the European Alps, using a 5-fold cross-validation setup, so we can use only the testing folds when later computing the area changes. 

Useful links:
- preprint: https://doi.org/10.22541/essoar.173557607.70204641/v1
- predicted glacier outlines: https://dcodrut.github.io/dl4gam_alps/
  - or here: https://huggingface.co/datasets/dcodrut/dl4gam_alps/viewer for sweeping over images + predictions using ↑↓
- dataset description: https://huggingface.co/datasets/dcodrut/dl4gam_alps
- model checkpoints: https://huggingface.co/dcodrut/dl4gam_alps

To reproduce the results, follow the steps below. Feel free to open an issue if you run into any problems.

### Set up the python environment using [mamba](https://github.com/conda-forge/miniforge)
```shell
mamba env create -f environment.yml
mamba activate dl4gam
cd dl4gam
```

### Reproduce the results for the European Alps
#### 1. Data downloading (inventory outlines and the processed NetCDF rasters):
- Download and process the glacier outlines: check `notebooks/process_outlines.ipynb`
- Download the glacier rasters:   
_(Note: each archive is ~10 GB per year — the inventory year & 2023. After extracting the NetCDF files, you’ll need ~24 GB per year.):_   
    ```shell
    bash ./scripts/s2_alps_plus/download_data.sh
    ```

#### 2. Data processing for training:
This includes patchifying (if configured), creating cross-validation splits, and computing data statistics for normalization:
```shell
python main_prep_data_train.py
```
By default, patches are sampled **on the fly** directly from the NetCDF rasters — no need to store them on disk.  
This is controlled by `SAMPLING_STEP_TRAIN` and `SAMPLING_STEP_VALID`. 
From the generated patches, we subsample `NUM_PATCHES_TRAIN` to increase ensemble diversity (although at the moment, patches are highly overlapping).

> 💡 Depending on your hardware, exporting patches to disk may speed up training. You can enable this in the config file.  
> Note: exported patches require ~50 GB of disk space.

#### 3. Model training, inference, calibration and area estimation:
- Train 10 U-Net models for each of the 5 cross-validation folds and infer on both years. Adjust the parameters based on the available hardware:   
```shell
python main_build_ensemble.py \
    --config_fp "./model_configs/s2_alps_plus/unet.yaml" \
    --n_splits 5 \
    --ensemble_size 10 \
    --n_gpus 4 \
    --max_tasks_per_gpu_train 2 \
    --max_tasks_per_gpu_infer 4 
```
- Calibrate the predictions for each ensemble member. After stacking, ensemble-averaged predictions are computed for both the inventory and 2023 imagery, followed by prediction interval estimation and calibration: 
```shell
for year in "inv" "2023"; do
    export S2_ALPS_YEAR=$year
    bash scripts/s2_alps_plus/eval.sh
    unset S2_ALPS_YEAR
done
```

#### 4. Post-processing:
- **Compute glacier area changes and uncertainties** using ensemble-averaged predictions:
```shell
python main_agg_stats.py \
  --model_dir "../data/external/_experiments/s2_alps_plus/unet" \
  --model_version "version_0" \
  --seed_list "all" \
  --split_list 1 2 3 4 5 \
  --subdir_list "inv" "2023" \
  --use_calib
```
- **Polygonize** the ensemble-averaged predictions:
```shell
for year in "inv" "2023"; do
    export S2_ALPS_YEAR=$year
    python main_polygonize.py \
      --model_dir "../data/external/_experiments/s2_alps_plus/unet" \
      --model_version "version_0" \
      --split_list 1 2 3 4 5 \
      --seed "all" \
      --use_calib
    unset S2_ALPS_YEAR
done
```
- **Visualize predictions** on top of imagery:
```shell
for year in "inv" "2023"; do
    export S2_ALPS_YEAR=$year
    python main_plot_imgs_with_preds.py \
      --plot_preds \
      --model_dir "../data/external/_experiments/s2_alps_plus/unet" \
      --model_version "version_0" \
      --seed "all" \
      --use_calib
    unset S2_ALPS_YEAR
done
```