from dataclasses import dataclass, field, MISSING
from typing import Tuple, Optional, Dict


@dataclass
class SMPModelConfig:
    """
    Base configuration for an SMP model.
    """
    name: str = MISSING  # https://github.com/qubvel-org/segmentation_models.pytorch/blob/ed2089f02cd34a0cbe7c0231324be4c8c7472aca/segmentation_models_pytorch/__init__.py#L26
    params: Dict = MISSING

    _target_: str = 'dl4gam.pl_modules.seg_model.SegModelSMP'

    @dataclass
    class InputSettings:
        # Inputs settings used in Diaconu et al. (2025)
        # TODO: use only the optical bands as default; make these settings external in a yaml file
        bands_input: Tuple[str] = field(default_factory=lambda: ('R', 'G', 'B', 'NIR', 'SWIR'))
        band_mask: str = 'mask_nok'  # this is computed automatically based on the QC band names from the dataset config
        dem: bool = True
        dhdt: bool = True
        optical_indices: Tuple[str] = field(default_factory=lambda: ('NDVI', 'NDWI', 'NDSI'))
        dem_features: Tuple[str] = field(
            default_factory=lambda: [
                'slope',
                'aspect_sin',
                'aspect_cos',
                'planform_curvature',
                'profile_curvature',
                'terrain_ruggedness_index'
            ])
        velocity: bool = False

    input_settings: InputSettings = field(default_factory=InputSettings)

    # Optional: path to external encoder weights
    encoder_weights_fp: Optional[str] = None


@dataclass
class UnetModelConfig(SMPModelConfig):
    """
    Default configuration for the model (UNet with all inputs, using SMP).
    """
    name: str = 'Unet'

    @dataclass
    class ModelArgs:
        encoder_name: str = 'resnet34'
        encoder_weights: str = 'imagenet'
        encoder_depth: int = 5
        activation: str = 'sigmoid'
        decoder_use_batchnorm: bool = False
        decoder_attention_type: Optional[str] = None

    params: ModelArgs = field(default_factory=ModelArgs)
