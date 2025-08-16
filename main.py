import logging
import logging.config
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig, SCMode

from dl4gam.configs.config import register_configs, DL4GAMCfg
from dl4gam.utils import parallel_utils

# Small utility to register a resolver for OmegaConf to get the parent directory of a path
# (depending on the stage, we use it to tell hydra to run in the parent directory of the current file to be produced)
OmegaConf.register_new_resolver("parentdir", lambda p: str(Path(p).parent))

# Strip all parts of a path to avoid issues with trailing slashes or spaces
OmegaConf.register_new_resolver("fpstrip", lambda p: str(Path(*[x.strip() for x in Path(p).parts])))

register_configs()


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg_dict: DictConfig):
    # Covert the DictConfig to our internal dataclass structure
    cfg: DL4GAMCfg = OmegaConf.to_container(
        cfg_dict,
        resolve=True,
        throw_on_missing=True,
        structured_config_mode=SCMode.INSTANTIATE,
    )

    # Check if the config has all the required parameters
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    log = logging.getLogger(__name__)

    # Show the hydra output directory
    hydra_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Hydra directory: {hydra_dir}")

    # Set the logging levels for external modules
    for module_name, log_level in cfg.external_modules_log_level.items():
        log.info(f"Setting logging level for module '{module_name}' to '{log_level}'")
        logging.getLogger(module_name).setLevel(log_level)

    # Set the number of processes and progress bar for parallel processing
    parallel_utils.set_default_num_procs(cfg.num_procs)
    parallel_utils.set_default_pbar(cfg.pbar)

    try:
        log.info(f"Stage to be executed: {cfg.stage}")

        # Get the parameters for the current stage to execute
        valid_stages = {
            attr.removeprefix('stage_') for attr in dir(cfg)
            if attr.startswith('stage_') and callable(getattr(cfg, attr))
        }
        if cfg.stage not in valid_stages:
            raise RuntimeError(f"Invalid stage={cfg.stage}; expected one of {sorted(valid_stages)}")

        # Run the stage function
        stage_func = getattr(cfg, f"stage_{cfg.stage}")
        stage_func()

    except KeyboardInterrupt:
        log.exception("Keyboard interrupt received, stopping the process.")
    except Exception:
        log.exception("An error occurred")


if __name__ == '__main__':
    main()
