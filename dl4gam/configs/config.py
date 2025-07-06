from dataclasses import dataclass, field
from typing import List, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from dl4gam.configs.datasets import S2AlpsConfig, S2AlpsPlusConfig
from dl4gam.configs.hydra import DL4GAMHydraConfig
from dl4gam.configs.models import UnetModelConfig
from dl4gam.configs.training import PLConfig


@dataclass
class DL4GAMConfig:
    """
    Full application config for the workflow.
    We set here the dataset, the model, and the settings for the workflow steps.
    """

    # Set defaults for: hydra, model, dataset and training setup
    defaults: List[Any] = field(default_factory=lambda: [
        '_self_',
        {'dataset': 's2_alps_plus'},
        {'model': 'unet'},
    ])

    # the following are pointers to the dataset and model configs
    dataset: Any = MISSING
    model: Any = MISSING

    # for the training setup we have a single config that is used for all models
    pl: PLConfig = field(default_factory=PLConfig)

    # Root working directory (everything is relative to this)
    working_dir: str = '../data/external/dl4gam'

    # Global settings
    min_glacier_area: float = 0.1  # km^2
    num_procs: int = 24  # for the steps that run in parallel
    pbar: bool = True  # whether to show progress bars when processing time-consuming steps
    preload_data: bool = False  # whether to load the netcdf files in memory before patchifying them

    # Workflow step configs: all the parameters are derived from the existing settings
    @property
    def step_process_inventory(self) -> dict:
        return {
            '_target_': 'dl4gam.workflow.process_inventory.main',
            'fp_in': self.dataset.outlines_fp,
            'fp_out': self.dataset.geoms_fp,
            'min_glacier_area': self.min_glacier_area,
            'buffers': self.dataset.buffers,
            'crs': self.dataset.crs,
            'gsd': self.dataset.gsd,
        }


# Register configs with Hydra's ConfigStore
def register_configs():
    cs = ConfigStore.instance()
    cs.store(group='dataset', name='s2_alps', node=S2AlpsConfig)
    cs.store(group='dataset', name='s2_alps_plus', node=S2AlpsPlusConfig)
    cs.store(group='model', name='unet', node=UnetModelConfig)
    cs.store(name='config', node=DL4GAMConfig)

    # Register the Hydra config separately in the hydra group
    cs.store(group="hydra", name="config", node=DL4GAMHydraConfig)
