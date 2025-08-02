import logging
import os

import pytorch_lightning as pl
from hydra.utils import instantiate

# https://github.com/PyTorchLightning/pytorch-lightning/issues/5225
if 'SLURM_NTASKS' in os.environ:
    del os.environ['SLURM_NTASKS']
if 'SLURM_JOB_NAME' in os.environ:
    del os.environ['SLURM_JOB_NAME']

log = logging.getLogger(__name__)


def main(**settings):
    # Fix the seed
    seed = settings['seed']
    log.info(f"Seeding everything to {seed}")
    pl.seed_everything(seed, workers=True)

    # Set up the tensorboard logger
    log.info(f"Setting up the TensorBoard logger: {settings['logger']}")
    tb_logger = instantiate(settings['logger'])

    # Data
    log.info(f"Instantiating the data module: {settings['data']}")
    dm = instantiate(settings['data'])

    # Model
    log.info(f"Instantiating the model: {settings['model']}")
    model = instantiate(settings['model'])

    # Task
    log.info(f"Instantiating the task: {settings['task']}")
    task = instantiate(settings['task'], model=model, _recursive_=False)

    # Callbacks
    log.info(f"Instantiating the model checkpoint callback: {settings['checkpoint_callback']}")
    checkpoint_callback = instantiate(settings['checkpoint_callback'])
    summary = pl.callbacks.ModelSummary(max_depth=-1)

    # Trainer
    log.info(f"Instantiating the trainer: {settings['trainer']}")
    trainer = instantiate(settings['trainer'], logger=tb_logger, callbacks=[checkpoint_callback, summary])

    # Finally, fit the model
    trainer.fit(task, dm)
    log.info(f'Best model {checkpoint_callback.best_model_path} score {checkpoint_callback.best_model_score}')
