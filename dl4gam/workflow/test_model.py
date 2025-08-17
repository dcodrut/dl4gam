import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from tqdm import tqdm

from dl4gam.lit.seg_task import GlSegTask

log = logging.getLogger(__name__)

from dl4gam.configs.config import SMPModelCfg, RunCfg, BaseDatasetCfg


def get_best_model_ckpt(checkpoint_dir, metric_name='val_loss_epoch', sort_method='min'):
    ckpt_list = sorted(list(checkpoint_dir.glob('*.ckpt')))
    if len(ckpt_list) == 0:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    ens_list = np.array([float(p.stem.split(f'{metric_name}=')[1]) for p in ckpt_list if metric_name in str(p)])

    # Get the index of the last best value
    if sort_method not in ['min', 'max']:
        raise ValueError(f"Invalid sort method: {sort_method}. Only 'min' or 'max' are supported.")
    sort_method_f = np.argmax if sort_method == 'max' else np.argmin
    i_best = len(ens_list) - sort_method_f(ens_list[::-1]) - 1
    ckpt_best = ckpt_list[i_best]

    return ckpt_best


def main(
        run_cfg: RunCfg,
        model_cfg: SMPModelCfg,
        dataset_cfg: BaseDatasetCfg,
        fold: str,
        checkpoint_dir: str | Path,
):
    # Prepare the checkpoint before anything else
    checkpoint_fp = get_best_model_ckpt(
        checkpoint_dir=checkpoint_dir,
        metric_name=run_cfg.checkpoint_callback.monitor,
        sort_method=run_cfg.checkpoint_callback.mode
    )

    # Data
    log.info(f"Instantiating the data module: {run_cfg.data}")
    dm = instantiate(run_cfg.data)
    dm.train_shuffle = False  # disable shuffling for the training dataloader

    # Model
    log.info(f"Instantiating the model: {model_cfg}")
    model = instantiate(model_cfg)

    # Task & weights loading
    device = f"cuda:{run_cfg.trainer.devices[0]}" if run_cfg.trainer.accelerator == 'gpu' else 'cpu'
    log.info(f"Loading the checkpoint {checkpoint_fp} on device {device}")
    task = GlSegTask.load_from_checkpoint(
        checkpoint_path=checkpoint_fp,
        map_location=device,
        model=model,
        **asdict(run_cfg.task)
    )

    # Trainer
    log.info(f"Instantiating the trainer: {run_cfg.trainer}")
    trainer = instantiate(run_cfg.trainer, logger=False)

    # Set the output directory
    ds = dataset_cfg  # TODO: allow different dataset than the one used for training
    task.outdir = checkpoint_fp.parent.parent / 'preds' / ds.name / ds.year / fold
    assert fold in ['train', 'val', 'test']
    log.info(f'Testing for fold = {fold}; results will be saved to {task.outdir}')

    # Get the list of glacier IDs to test on
    log.info(f"Reading the glacier IDs from {ds.split_csv} for fold {fold}")
    split_df = pd.read_csv(ds.split_csv, index_col='entry_id')
    glacier_ids_crt_fold = split_df[split_df['fold'] == fold].index.tolist()

    cubes_dir = Path(ds.cubes_dir)
    fp_cubes = list(cubes_dir.rglob('*.nc'))
    glacier_ids_crt_dir = set([p.parent.name for p in fp_cubes])
    log.info(f"Found {len(fp_cubes)} cubes in {cubes_dir}")

    glacier_ids_final = sorted(set(glacier_ids_crt_dir) & set(glacier_ids_crt_fold))
    log.info(f'#glaciers to test on = {len(glacier_ids_final)}')

    fp_cubes = list(filter(lambda x: x.parent.name in glacier_ids_final, fp_cubes))
    if len(fp_cubes) != len(glacier_ids_final):
        raise ValueError(
            f"We expect one netCDF file per glacier, "
            f"but found a mismatch: {len(fp_cubes)} files for {len(glacier_ids_final)} glaciers. "
            f"Check the split CSV {ds.split_csv} and the netCDF files in {cubes_dir}."
        )

    dl_list = dm.test_dataloaders_per_glacier(fp_rasters=fp_cubes)
    times = {}  # count the time for each glacier
    for dl in tqdm(dl_list, desc='Testing per glacier'):
        start_time = time.time()
        trainer.test(model=task, dataloaders=dl)
        times[dl.dataset.fp.parent.name] = time.time() - start_time
    times_df = pd.DataFrame.from_dict(times, orient='index', columns=['time'])
    times_df.index.name = 'entry_id'
    log.info(f"Time per glacier (decreasing):\n{times_df.sort_values('time', ascending=False)}")
    log.info(f"Time per glacier stats:\n{times_df.describe()}")
    log.info(f"Total time: {times_df['time'].sum() / 3600:.2f} h")

    # Save the time list
    time_list_fp = task.outdir.parent / f'infer_time_per_g_{fold}.csv'
    times_df.to_csv(time_list_fp)
    log.info(f"Time list saved to {time_list_fp}")
