from dataclasses import dataclass, field
from typing import Tuple, Optional

from omegaconf import MISSING


@dataclass
class InputConfig:
    """
    Channels and auxiliary features to feed into the segmentation model.
    """
    # Spectral bands
    bands_input: Tuple[str, ...] = MISSING  # e.g. ('R', 'G', 'B', 'NIR', 'SWIR')

    # Mask band for masking out e.g. cloudy pixels; can be None otherwise this band has to be present in the dataset
    # Note that we will always mask out the NA values in the input bands (otherwise the training will fail)
    band_mask: Optional[str] = 'mask_nok'  # this is computed automatically if data is downloaded with our GEE pipeline

    # Indices derived from optical bands (will be computed on the fly; make sure the required bands are present)
    optical_indices: Optional[Tuple[str, ...]] = None  # subset of ('NDVI', 'NDWI', 'NDSI')

    # Auxiliary input rasters (see `extra_rasters` from the dataset config)
    # TODO: combine these and the dem_features into a single list of features
    dem: bool = False
    dhdt: bool = False
    velocity: bool = False

    # Topographic features derived from DEM (see `xdem_features` from the dataset config)
    dem_features: Optional[Tuple[str, ...]] = None


@dataclass
class SMPModelConfig:
    """
    Base config for any SMP-based segmentation model.
    """
    _target_: str = 'dl4gam.pl_modules.seg_model.SegModelSMP'

    # Name of the SMP model (e.g., 'Unet', 'DeepLabV3')
    # https://github.com/qubvel-org/segmentation_models.pytorch/blob/ed2089f02cd34a0cbe7c0231324be4c8c7472aca/segmentation_models_pytorch/__init__.py#L26
    name: str = MISSING

    @dataclass
    class ModelArgs:
        """
        Hyperparameters for the segmentation-model architecture. Default from Diaconu et al. 2025 - DL4GAM.
        """
        encoder_name: str = MISSING
        encoder_weights: Optional[str] = MISSING
        encoder_depth: int = MISSING
        activation: str = 'sigmoid'
        decoder_use_batchnorm: bool = False
        decoder_attention_type: Optional[str] = None

    # Architecture-specific hyperparameters
    params: ModelArgs = field(default_factory=ModelArgs)

    # Input channels and auxiliary feature settings
    input_settings: InputConfig = MISSING

    # Optional path to external encoder weights
    encoder_weights_fp: Optional[str] = None
