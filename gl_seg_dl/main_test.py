from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
from tqdm import tqdm
import logging

# local imports
from config import C
import models
from task.data import GlSegDataModule
from task.seg import GlSegTask
from utils.general import str2bool

# Logger (console and TensorBoard)
root_logger = logging.getLogger('pytorch_lightning')
root_logger.setLevel(logging.INFO)
fmt = '[%(levelname)s] - %(asctime)s - %(name)s: %(message)s (%(filename)s:%(funcName)s:%(lineno)d)'
root_logger.handlers[0].setFormatter(logging.Formatter(fmt))
logger = logging.getLogger('pytorch_lightning.core')


def test_model(
        settings, fold, test_per_glacier=False, glacier_id_list=None, checkpoint=None, patch_radius=None,
        sampling_step=None, preload_data=None
):
    logger.info(f'Settings: {settings}')

    # Data
    data_params = settings['data']
    data_params['input_settings'] = settings['model']['inputs']
    dm = GlSegDataModule(**data_params)
    dm.train_shuffle = False  # disable shuffling for the training dataloader

    # Model & task
    model_class = getattr(models, settings['model']['class'])
    model = model_class(
        input_settings=settings['model']['inputs'] if 'inputs' in settings['model'] else None,
        other_settings=settings['model']['other'] if 'other' in settings['model'] else None,
        model_name=settings['model']['name'],
        model_args=settings['model']['args'],
    )
    logger.info(f'Loading model from {checkpoint}')
    task_params = settings['task']
    device = f"cuda:{settings['trainer']['devices'][0]}" if settings['trainer']['accelerator'] == 'gpu' else 'cpu'
    task = GlSegTask.load_from_checkpoint(
        checkpoint_path=checkpoint,
        map_location=device,
        model=model,
        task_params=task_params
    )

    # Trainer
    trainer_dict = settings['trainer']
    trainer = pl.Trainer(**trainer_dict, logger=False)

    logger.info(f'test_per_glacier = {test_per_glacier}')
    checkpoint_root_dir = Path(checkpoint).parent.parent
    if test_per_glacier:
        p = list(Path(data_params['rasters_dir']).parts)
        ds_name = p[p.index('glacier_wide') - 2]
        subdir = p[p.index('glacier_wide') - 1]
        root_outdir = checkpoint_root_dir / 'output' / 'preds' / ds_name / subdir
    else:
        p = list(Path(data_params['data_root_dir']).parts)
        ds_name = p[p.index('patches') + 1]
        root_outdir = checkpoint_root_dir / 'output' / 'stats' / f"patches_{ds_name}"

    assert fold in ['s_train', 's_valid', 's_test']
    logger.info(f'Testing for fold = {fold}')

    task.outdir = root_outdir / fold
    logger.info(f"output directory = {task.outdir}")
    if not test_per_glacier:
        assert fold in ('s_train', 's_valid', 's_test')
        dm.setup('test' if fold == 's_test' else 'fit')
        if fold == 's_train':
            dl = dm.train_dataloader()
        elif fold == 's_valid':
            dl = dm.val_dataloader()
        else:
            dl = dm.test_dataloader()

        results = trainer.validate(model=task, dataloaders=dl)
        logger.info(f'results = {results}')
    else:
        dir_fp = Path(dm.rasters_dir)
        logger.info(f'Reading the glaciers IDs based on the rasters from {dir_fp}')
        fp_list = list(dir_fp.glob('**/*.nc'))
        glacier_id_list_crt_dir = set([p.parent.name for p in fp_list])
        logger.info(f'#glaciers in the current rasters dir = {len(glacier_id_list_crt_dir)}')

        glacier_id_list_final = sorted(set(glacier_id_list_crt_dir) & set(glacier_id_list))
        logger.info(f'#glaciers to test on = {len(glacier_id_list_final)}')

        # TODO: parallelize this and avoid loading everything in-memory at the beginning
        dl_list = dm.test_dataloaders_per_glacier(
            gid_list=glacier_id_list_final,
            patch_radius=patch_radius,
            sampling_step=sampling_step,
            preload_data=preload_data
        )
        for dl in tqdm(dl_list, desc='Testing per glacier'):
            trainer.test(model=task, dataloaders=dl)


def get_best_model_ckpt(checkpoint_dir, metric_name='val_loss_epoch', sort_method='min'):
    checkpoint_dir = Path(checkpoint_dir)
    assert checkpoint_dir.exists(), f'{checkpoint_dir} not found'
    assert sort_method in ('max', 'min')

    ckpt_list = sorted(list(checkpoint_dir.glob('*.ckpt')))
    ens_list = np.array([float(p.stem.split(f'{metric_name}=')[1]) for p in ckpt_list if metric_name in str(p)])

    # get the index of the last best value
    sort_method_f = np.argmax if sort_method == 'max' else np.argmin
    i_best = len(ens_list) - sort_method_f(ens_list[::-1]) - 1
    ckpt_best = ckpt_list[i_best]

    return ckpt_best


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--settings_fp', type=str, metavar='path/to/settings.yaml', help='yaml with all the settings')
    parser.add_argument('--checkpoint', type=str, metavar='path/to/checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--checkpoint_dir', type=str, metavar='path/to/checkpoint_dir',
                        help='a directory from which the model with the best evaluation score will be selected '
                             '(alternative to checkpoint_file)', default=None)
    parser.add_argument('--fold', type=str, metavar='s_train|s_valid|s_test', required=True,
                        help='which subset to test on: either s_train, s_valid or s_test')
    parser.add_argument('--test_per_glacier', type=str2bool, required=True,
                        help='whether to apply the model separately for each glacier instead of using the patches'
                             '(by generating in-memory all the patches)')
    parser.add_argument('--rasters_dir', type=str, required=False,
                        help='directory on which to test the model, for the case when test_per_glacier is True; '
                             'if not provided, the one from the config file is used'
                        )
    parser.add_argument('--split_fp', type=str, required=False,
                        help='path to a (geo)pandas dataframe containing the list of glaciers to be tested '
                             'and the corresponding fold of each split. Either .csv or .shp.')
    parser.add_argument('--gpu_id', type=int, default=None, required=False
                        , help='GPU ID (if not given, the one from the config is used)'
                        )
    args = parser.parse_args()

    # prepare the checkpoint
    if args.checkpoint_dir is not None:
        # get the best checkpoint
        checkpoint_file = get_best_model_ckpt(
            checkpoint_dir=args.checkpoint_dir,
            metric_name='w_JaccardIndex_val_epoch_avg_per_g',
            sort_method='max'
        )
    else:
        checkpoint_file = args.checkpoint

    # get the settings (assuming it was saved in the model's results directory if not given)
    if args.settings_fp is None:
        model_dir = Path(checkpoint_file).parent.parent
        settings_fp = model_dir / 'settings.yaml'
    else:
        settings_fp = args.settings_fp

    with open(settings_fp, 'r') as fp:
        all_settings = yaml.load(fp, Loader=yaml.FullLoader)

    # overwrite the gpu id if provided
    if args.gpu_id is not None:
        all_settings['trainer']['devices'] = [args.gpu_id]

    # set the raster directory to the command line argument if given, otherwise use the inference dirs from the config
    rasters_dir = args.rasters_dir if args.rasters_dir is not None else C.DIR_GL_INVENTORY
    assert Path(rasters_dir).exists(), f"rasters_dir = {rasters_dir} does not exist"
    all_settings['data']['rasters_dir'] = rasters_dir

    # choose the glaciers for the specified fold using the split csv file (use the cl-one if provided)
    if args.split_fp is not None:
        split_fp = Path(args.split_fp)
    else:
        split_fp = Path(all_settings['data']['all_splits_fp'])
    logger.info(f"Reading the split dataframe from {split_fp}")
    split_df = pd.read_csv(split_fp)

    # get the split on which the current model was trained on
    split_name = f"split_{str(checkpoint_file).split('split_')[1].split('/')[0]}"

    # get the list of glaciers for the specified fold
    fold_name = f"fold_{args.fold[2:]}"
    glacier_ids = sorted(list(split_df[split_df[split_name] == fold_name].entry_id))
    logger.info(f"split = {split_name}; fold = {fold_name}; #glaciers = {len(glacier_ids)}")

    test_model(
        settings=all_settings,
        checkpoint=checkpoint_file,
        test_per_glacier=args.test_per_glacier,
        glacier_id_list=glacier_ids,
        fold=args.fold,
        patch_radius=C.PATCH_RADIUS,
        sampling_step=C.SAMPLING_STEP_INFER,
        preload_data=C.PRELOAD_DATA_INFER
    )
