import logging
from pathlib import Path

import segmentation_models_pytorch as smp
import torch


class SegModel(torch.nn.Module):
    def __init__(self, input_settings, other_settings, model_name, model_args):
        super().__init__()
        self.model_args = model_args
        self.model_name = model_name

        # extract the inputs
        self.input_settings = input_settings
        self.bands = input_settings['bands_input']
        self.use_optical_indices = input_settings['optical_indices']
        self.use_dem = input_settings['dem']
        self.use_dem_features = input_settings['dem_features']
        self.use_dhdt = input_settings['dhdt']
        self.use_velocities = input_settings['velocity']

        # prepare the logger
        self.logger = logging.getLogger('pytorch_lightning.core')

        # compute the number of input channels based on what variables are used
        num_ch = len(self.bands)
        num_ch += 3 * self.use_optical_indices
        num_ch += 1 * self.use_dem
        num_ch += 6 * self.use_dem_features
        num_ch += 1 * self.use_dhdt
        num_ch += 1 * self.use_velocities
        self.model_args['in_channels'] = num_ch

        # set the number of output channels
        self.model_args['classes'] = 1

        self.logger.info(f'Building Unet with {self.model_args}')
        self.seg_model = getattr(smp, self.model_name)(**self.model_args)

        if other_settings is not None:
            use_external_weights = other_settings['external_encoder_weights'] is not None
            # ensure only one set of pretrained weights are used
            assert self.model_args['encoder_weights'] is None or not use_external_weights, \
                'Choose whether to use smp provided weights or external ones, not both'

            # load the external weights
            if use_external_weights:
                fp = other_settings['external_encoder_weights']
                checkpoint = torch.load(fp, map_location='cpu')
                state_dict = checkpoint['state_dict']
                if Path(fp).stem == 'B13_rn50_moco_0099':
                    # name corrections: https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/benchmark/transfer_classification/linear_BE_moco.py#L251
                    for k in list(state_dict.keys()):
                        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                        del state_dict[k]

                # discard the input bands which are not needed
                band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
                idx_bands = [band_names.index(b) for b in input_settings['bands_input']]

                # use one more band for the DEM if needed
                if self.use_dem:
                    idx_bands.append(band_names.index('B8A'))

                state_dict['conv1.weight'] = state_dict['conv1.weight'][:, idx_bands, ...]
                self.seg_model.encoder.load_state_dict(state_dict)

    def forward(self, batch):
        input_list = []

        # add the S2-bands
        input_list.append(batch['band_data'])

        # add the indices if needed
        if self.use_optical_indices:
            input_list.append(batch['ndsi'][:, None, :, :])
            input_list.append(batch['ndvi'][:, None, :, :])
            input_list.append(batch['ndwi'][:, None, :, :])

        # add the DEM if needed
        if self.use_dem:
            input_list.append(batch['dem'][:, None, :, :])

        # add the DEM features if needed
        if self.use_dem_features:
            for k in [
                'slope',
                'aspect_sin',
                'aspect_cos',
                'planform_curvature',
                'profile_curvature',
                'terrain_ruggedness_index'
            ]:
                input_list.append(batch[k][:, None, :, :])

        # add the dhdt if needed
        if self.use_dhdt:
            input_list.append(batch['dhdt'][:, None, :, :])

        # add the velocities if needed
        if self.use_velocities:
            input_list.append(batch['v'][:, None, :, :])

        # concatenate all the inputs over channel
        inputs = torch.cat(input_list, dim=1)

        # get the predictions
        preds = self.seg_model(inputs)

        return preds
