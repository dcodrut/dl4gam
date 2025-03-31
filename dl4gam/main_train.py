import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers
import yaml

# local imports
import config
import models
from task.data import GlSegDataModule
from task.seg import GlSegTask

# https://github.com/PyTorchLightning/pytorch-lightning/issues/5225
if 'SLURM_NTASKS' in os.environ:
    del os.environ['SLURM_NTASKS']
if 'SLURM_JOB_NAME' in os.environ:
    del os.environ['SLURM_JOB_NAME']


def train_model(settings: dict, patches_on_disk: bool = False, n_patches: int = None):
    # Logger (console and TensorBoard)
    tb_logger = pl.loggers.TensorBoardLogger(**settings['logger'])
    Path(tb_logger.log_dir).mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger('pytorch_lightning')
    root_logger.setLevel(logging.INFO)
    fmt = '[%(levelname)s] - %(asctime)s - %(name)s: %(message)s (%(filename)s:%(funcName)s:%(lineno)d)'
    root_logger.handlers[0].setFormatter(logging.Formatter(fmt))
    logger = logging.getLogger('pytorch_lightning.core')
    fh = logging.FileHandler(Path(tb_logger.log_dir, 'console.log'))
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    # fix the seed
    seed = settings['task']['seed']
    pl.seed_everything(seed, workers=True)
    logger.info(f'Initial settings: {settings}')

    # Data
    data_params = settings['data'].copy()
    data_params['input_settings'] = settings['model']['inputs']
    dm = GlSegDataModule(**data_params)

    # Model
    model_class = getattr(models, settings['model']['class'])
    model = model_class(
        input_settings=settings['model']['inputs'] if 'inputs' in settings['model'] else None,
        other_settings=settings['model']['other'] if 'other' in settings['model'] else None,
        model_args=settings['model']['args'],
        model_name=settings['model']['name']
    )

    # Task
    task_params = settings['task']
    task = GlSegTask(model=model, task_params=task_params)

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='w_JaccardIndex_val_epoch_avg_per_g',
        filename='ckpt-{epoch:02d}-{w_JaccardIndex_val_epoch_avg_per_g:.4f}',
        save_top_k=1,
        save_last=True,
        mode='max',
        every_n_epochs=1
    )
    summary = pl.callbacks.ModelSummary(max_depth=-1)

    # Trainer
    trainer_dict = settings['trainer']
    trainer = pl.Trainer(logger=tb_logger, callbacks=[checkpoint_callback, summary], **trainer_dict)

    # Save the config file, after adding the slurm job id (if exists)
    path_out = Path(tb_logger.log_dir, 'settings.yaml')
    setting_dict_save = settings.copy()
    setting_dict_save['SLURM_JOBID'] = os.environ['SLURM_JOBID'] if 'SLURM_JOBID' in os.environ else None
    with open(path_out, 'w') as fp:
        yaml.dump(setting_dict_save, fp, sort_keys=False)
    logger.info(f'Settings:\n{json.dumps(settings, sort_keys=False, indent=4)}')
    logger.info(f'Settings exported to {path_out}')

    # explicitly call the setup method of the DataModule so we can print the data stats
    dm.setup(stage='fit')
    for ds_name in ['train', 'valid']:
        ds = getattr(dm, f'{ds_name}_ds')
        if patches_on_disk:
            logger.info(f'{ds_name} dataset: {len(ds)} patches from {ds.n_glaciers} glaciers')
        else:
            # this means ds is a ConcatDataset
            ds_sizes = [len(x) for x in ds.datasets]
            logger.info(
                f'{ds_name} dataset: '
                f'{len(ds)} patches from {len(ds_sizes)} glaciers; '
                f'#samples per glacier: min = {min(ds_sizes)} max = {max(ds_sizes)} avg = {np.mean(ds_sizes):.1f}'
            )

            # subsample the training set if needed
            if ds_name == 'train' and n_patches is not None:
                logger.info(f'Subsampling the training set to {n_patches} patches')
                dm.subsample_train_ds(n_patches, seed=seed)
                ds_sizes = [len(x) for x in dm.train_ds.datasets]
                logger.info(
                    f'{ds_name} dataset: '
                    f'{len(dm.train_ds)} patches from {len(ds_sizes)} glaciers; '
                    f'#samples per glacier: min = {min(ds_sizes)} max = {max(ds_sizes)} avg = {np.mean(ds_sizes):.1f}'
                )

    trainer.fit(task, dm)
    logger.info(f'Best model {checkpoint_callback.best_model_path} with score {checkpoint_callback.best_model_score}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--config_fp',
        type=str, metavar='path/to/settings.yaml', help='yaml with all the model settings', required=True)
    parser.add_argument(
        '--seed',
        type=int, help='training seed (if not given, the one from the config is used)', default=None, required=False)
    parser.add_argument(
        '--split',
        type=str, help='training split (if not given, the one from the config is used)', default=None, required=False,
    )
    parser.add_argument(
        '--gpu_id',
        type=int, help='GPU ID (if not given, the one from the config is used)', default=None, required=False)
    args = parser.parse_args()

    # read the settings
    assert Path(args.config_fp).exists(), f'Settings file ({args.config_fp}) not found.'
    with open(args.config_fp, 'r') as f:
        all_settings = yaml.load(f, Loader=yaml.FullLoader)

    # add the patch sampling parameters if needed
    # (we don't store them in the model config file because when patches are exported to disk, we need the sampling
    #  parameters in main_prep_data_train.py)

    # get the constant class
    config_cls_name = all_settings['config']
    C = getattr(config, config_cls_name)

    # fill in the missing parameters
    all_settings['data']['rasters_dir'] = str(C.DIR_GL_RASTERS)
    all_settings['data']['all_splits_fp'] = str(Path(C.DIR_OUTLINES_SPLIT) / 'map_all_splits_all_folds.csv')

    if not C.EXPORT_PATCHES:
        all_settings['data']['patch_radius'] = C.PATCH_RADIUS
        all_settings['data']['sampling_step_train'] = C.SAMPLING_STEP_TRAIN
        all_settings['data']['sampling_step_valid'] = C.SAMPLING_STEP_VALID
        all_settings['data']['preload_data'] = C.PRELOAD_DATA

        if C.SAMPLE_PATCHES_EACH_EPOCH:
            # limit the number of batches in the training set s.t. we sample different patches each epoch
            # (with shuffle=True and a large enough number of patches, the epoch will be different each time)
            num_batches = C.NUM_PATCHES_TRAIN // all_settings['data']['train_batch_size']
            all_settings['trainer']['limit_train_batches'] = num_batches
    else:
        # set the patch directory
        all_settings['data']['patches_dir'] = str(C.DIR_GL_PATCHES)

    # for testing, we always build up the patches in memory
    all_settings['data']['sampling_step_test'] = C.SAMPLING_STEP_TEST  # will be used for inference in the future

    # overwrite the seed if provided
    if args.seed is not None:
        all_settings['task']['seed'] = args.seed

    # overwrite the split if provided
    if args.split is not None:
        all_settings['data']['split'] = args.split
        all_settings['data']['data_stats_fp'] = str(C.NORM_STATS_DIR / f'stats_agg_{args.split}.csv')
        all_settings['logger']['name'] = str(Path(all_settings['logger']['name']).parent / args.split)

    # add the seed as a subfolder
    all_settings['logger']['name'] = str(Path(all_settings['logger']['name']) / f'seed_{all_settings["task"]["seed"]}')

    # overwrite the gpu ID if provided
    if args.gpu_id is not None:
        all_settings['trainer']['devices'] = [args.gpu_id]

    train_model(
        all_settings,
        patches_on_disk=C.EXPORT_PATCHES,
        n_patches=C.NUM_PATCHES_TRAIN if not C.SAMPLE_PATCHES_EACH_EPOCH else None
    )
