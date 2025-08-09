import functools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import xarray as xr
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from dl4gam.utils import run_in_parallel, get_patches_df

log = logging.getLogger(__name__)


def extract_inputs(ds, fp, input_settings):
    band_names = ds.band_data.attrs['long_name']
    assert set(input_settings['bands_input']).issubset(band_names), \
        f"Invalid bands: {input_settings['bands_input']} not in {band_names} for {fp}"

    idx_bands = [band_names.index(b) for b in input_settings['bands_input']]
    band_data = ds.band_data.isel(band=idx_bands).values.astype(np.float32)

    # prepare the mask
    mask_name = input_settings['band_mask']
    if mask_name is not None:
        mask_no_data = (ds[mask_name].values == 1)
    else:
        mask_no_data = np.zeros_like(ds.band_data.isel(band=0).values).astype(bool)

    # include to the nodata mask any NaN pixel
    # (it happened once that a few pixels were missing only from the last band but the mask did not include them)
    mask_na_per_band = np.isnan(band_data)
    if mask_na_per_band.sum() > 0:
        idx_na = np.where(mask_na_per_band)

        # fill in the gaps with the average
        avg_per_band = np.nansum(np.nansum(band_data, axis=-1), axis=-1) / np.prod(band_data.shape[-2:])
        band_data[idx_na[0], idx_na[1], idx_na[2]] = avg_per_band[idx_na[0]]

        # make sure that these pixels are masked in mask_no_data too
        mask_na = mask_na_per_band.any(axis=0)
        mask_no_data |= mask_na

    data = {
        'band_data': band_data,
        'mask_no_data': mask_no_data,
        'mask_crt_g': ds.mask_glacier.values == 1,
        'mask_all_g': ~np.isnan(ds.mask_all_glaciers_id.values),
        'fp': str(fp),
        'glacier_area': ds.attrs['glacier_area']
    }

    # add the debris mask if available
    if 'mask_debris' in ds.data_vars:
        mask_debris_crt_g = (ds.mask_debris.values == 1) & data['mask_crt_g']
        mask_debris_all_g = (ds.mask_debris.values == 1) & data['mask_all_g']
        data['mask_debris_crt_g'] = mask_debris_crt_g
        data['mask_debris_all_g'] = mask_debris_all_g

    if input_settings['dem']:
        dem = ds.dem.values.astype(np.float32)
        # fill in the NAs with the average
        dem[np.isnan(dem)] = np.mean(dem[~np.isnan(dem)])
        data['dem'] = dem

    if input_settings['dhdt']:
        dhdt = ds.dhdt.values.astype(np.float32)
        # fill in the NAs with zeros
        dhdt[np.isnan(dhdt)] = 0.0
        data['dhdt'] = dhdt

    if input_settings['velocity']:
        v = ds.v.values.astype(np.float32)
        # fill in the NAs with the average
        v[np.isnan(v)] = np.mean(v[~np.isnan(v)])
        data['v'] = v

    if input_settings['optical_indices']:
        # compute the provided indices (has to be a subset of ['NDSI', 'NDVI', 'NDWI'])
        assert set(input_settings['optical_indices']).issubset({'NDSI', 'NDVI', 'NDWI'})

        # NDSI = (Green - SWIR) / (Green + SWIR)
        if 'NDSI' in input_settings['optical_indices']:
            g = band_data[input_settings['bands_input'].index('G')]
            swir = band_data[input_settings['bands_input'].index('SWIR')]
            den = g + swir
            den[den == 0] = 1  # avoid division by zero
            data['NDSI'] = (g - swir) / den

        # NDVI = (NIR - Red) / (NIR + Red)
        if 'NDVI' in input_settings['optical_indices']:
            r = band_data[input_settings['bands_input'].index('R')]
            nir = band_data[input_settings['bands_input'].index('NIR')]
            den = nir + r
            den[den == 0] = 1  # avoid division by zero
            data['NDVI'] = (nir - r) / den

        # NDWI = (Green - NIR) / (Green + NIR)
        if 'NDWI' in input_settings['optical_indices']:
            g = band_data[input_settings['bands_input'].index('G')]
            nir = band_data[input_settings['bands_input'].index('NIR')]
            den = g + nir
            den[den == 0] = 1  # avoid division by zero
            data['NDWI'] = (g - nir) / den

    if input_settings['dem_features']:
        # the features should be a subset of
        # ['slope', 'aspect_sin', 'aspect_cos', 'planform_curvature', 'profile_curvature', 'terrain_ruggedness_index']
        assert set(input_settings['dem_features']).issubset({
            'slope', 'aspect_sin', 'aspect_cos', 'planform_curvature', 'profile_curvature', 'terrain_ruggedness_index'
        })

        if 'slope' in input_settings['dem_features']:
            data['slope'] = ds.slope.values.astype(np.float32) / 90.  # scale the slope to [0, 1]

        # compute the sine and cosine of the aspect
        # TODO: maybe remove when using geometric augmentation as the aspect doesn't physically make sense anymore)
        if 'aspect_sin' in input_settings['dem_features']:
            data['aspect_sin'] = np.sin(ds.aspect.values.astype(np.float32) * np.pi / 180)
        if 'aspect_cos' in input_settings['dem_features']:
            data['aspect_cos'] = np.cos(ds.aspect.values.astype(np.float32) * np.pi / 180)

        # add the planform curvature, profile curvature, terrain ruggedness index, which will be later normalized
        for k in ['planform_curvature', 'profile_curvature', 'terrain_ruggedness_index']:
            if k in input_settings['dem_features']:
                data[k] = ds[k].values.astype(np.float32)

    return data


def standardize_inputs(data, stats_df, scale_each_band, bands_input):
    band_data_sdf = pd.concat([stats_df[stats_df.var_name == b] for b in bands_input])
    mu = band_data_sdf.mu.values
    stddev = band_data_sdf.stddev.values

    if not scale_each_band:
        mu[:] = mu.mean()
        stddev[:] = stddev.mean()

    data['band_data'] -= mu[:, None, None]
    data['band_data'] /= stddev[:, None, None]

    # do the same for the static variables that need to be standardized
    for v in ['dem', 'dhdt', 'v', 'planform_curvature', 'profile_curvature', 'terrain_ruggedness_index']:
        if v in data:
            sdf = stats_df[stats_df.var_name == v]
            assert len(sdf) == 1, f"Expecting one stats row for {v}"
            mu = sdf.mu.values[0]
            stddev = sdf.stddev.values[0]
            data[v] -= mu
            data[v] /= stddev


def minmax_scale_inputs(data, stats_df, scale_each_band, bands_input):
    band_data_sdf = pd.concat([stats_df[stats_df.var_name == b] for b in bands_input])
    vmin = band_data_sdf.vmin.values
    vmax = band_data_sdf.vmax.values

    # clip the values to the min and max
    dtype = data['band_data'].dtype
    data['band_data'] = np.clip(data['band_data'], vmin[:, None, None], vmax[:, None, None]).astype(dtype)

    # scale to [0, 1]
    if not scale_each_band:
        vmin[:] = vmin.min()
        vmax[:] = vmax.max()
    data['band_data'] -= vmin[:, None, None]
    data['band_data'] /= (vmax[:, None, None] - vmin[:, None, None])

    # do the same for the static variables that need to be normalized
    for v in ['dem', 'dhdt', 'v', 'planform_curvature', 'profile_curvature', 'terrain_ruggedness_index']:
        if v in data:
            # apply the scaling
            sdf = stats_df[stats_df.var_name == v]
            assert len(sdf) == 1, f"Expecting one stats row for {v}"
            vmin = sdf.vmin.values[0]
            vmax = sdf.vmax.values[0]

            # clip the values to the min and max
            data[v] = np.clip(data[v], vmin, vmax).astype(dtype)

            # scale to [0, 1]
            data[v] -= vmin
            data[v] /= (vmax - vmin)


class GlSegPatchDataset(Dataset):
    def __init__(self, input_settings, folder=None, fp_list=None, standardize_data=False, minmax_scale_data=False,
                 scale_each_band=True, data_stats_df=None, use_augmentation=False):
        assert folder is not None or fp_list is not None

        if folder is not None:
            folder = Path(folder)
            self.fp_list = sorted(list(folder.rglob('*.nc')))

            if len(self.fp_list) == 0:
                raise FileNotFoundError(f'No .nc files found in the folder: {str(folder)}')
        else:
            self.fp_list = fp_list

        self.input_settings = input_settings
        self.standardize_data = standardize_data
        self.minmax_scale_data = minmax_scale_data
        self.scale_each_band = scale_each_band
        self.data_stats_df = data_stats_df

        # save the glaciers IDs
        self.glaciers = sorted(list(set([fp.parent.name for fp in self.fp_list])))
        self.n_glaciers = len(self.glaciers)

        if use_augmentation:
            # D4: https://albumentations.ai/docs/api_reference/full_reference/?h=d4#albumentations.augmentations.geometric.transforms.D4
            self.aug_transforms = [
                # rotate_90 (note that transpose() is doing a flip along the main diagonal)
                lambda ds: ds.transpose('band', 'x', 'y').isel(y=slice(None, None, -1)),
                # rotate_180, i.e. flip_horizontal + flip_vertical
                lambda ds: ds.isel(x=slice(None, None, -1), y=slice(None, None, -1)),
                # rotate_270
                lambda ds: ds.transpose('band', 'x', 'y').isel(x=slice(None, None, -1)),
                # flip_horizontal
                lambda ds: ds.isel(x=slice(None, None, -1)),
                # flip_vertical
                lambda ds: ds.isel(y=slice(None, None, -1)),
                # flip along the main diagonal
                lambda ds: ds.transpose('band', 'x', 'y'),
                # flip along the counter diagonal (i.e. previous + rotate_180)
                lambda ds: ds.transpose('band', 'x', 'y').isel(x=slice(None, None, -1), y=slice(None, None, -1))
            ]
        else:
            self.aug_transforms = None

    def process_data(self, ds, fp):
        # apply one of the augmentations if needed
        # (directly on the xarray dataset s.t. all the variables are transformed)
        if self.aug_transforms is not None and np.random.rand() < 0.5:
            i = np.random.randint(0, len(self.aug_transforms))
            ds = self.aug_transforms[i](ds)

        # extract the inputs
        data = extract_inputs(ds=ds, fp=fp, input_settings=self.input_settings)

        # standardize/scale the inputs if needed
        if self.standardize_data and self.minmax_scale_data:
            raise ValueError(
                'Cannot standardize and min-max scale the data at the same time. '
                'Please choose one of the two options.'
            )

        if self.standardize_data:
            standardize_inputs(
                data,
                stats_df=self.data_stats_df,
                scale_each_band=self.scale_each_band,
                bands_input=self.input_settings['bands_input']
            )
        if self.minmax_scale_data:
            minmax_scale_inputs(
                data,
                stats_df=self.data_stats_df,
                scale_each_band=self.scale_each_band,
                bands_input=self.input_settings['bands_input']
            )

        return data

    def __getitem__(self, idx):
        # read the current file
        fp = self.fp_list[idx]
        nc = xr.open_dataset(fp, decode_coords='all')

        data = self.process_data(ds=nc, fp=fp)

        return data

    def __len__(self):
        return len(self.fp_list)


class GlSegDataset(GlSegPatchDataset):
    def __init__(self, fp, patch_radius, stride=None, preload_data=False, add_extremes=False, **kwargs):
        self.fp = fp
        super().__init__(folder=None, fp_list=[fp], **kwargs)
        self.patch_radius = patch_radius

        # get all possible patches for the glacier
        if preload_data:
            self.nc = xr.load_dataset(fp, decode_coords='all')
        else:
            self.nc = xr.open_dataset(fp, decode_coords='all')
        self.patches_df = get_patches_df(
            self.nc,
            patch_radius=patch_radius,
            stride=stride,
            add_center=False,
            add_centroid=True,
            add_extremes=add_extremes
        )

    def subsample(self, n, seed, with_replacement=False):
        # Subsample the patches
        if n > len(self.patches_df):
            log.warning(f'Not enough patches to sample {n} from {len(self.patches_df)}; keeping all patches')
            return

        self.patches_df = self.patches_df.sample(n, replace=with_replacement, random_state=seed)

    def __getitem__(self, idx):
        r = self.patches_df.iloc[idx]
        nc_patch = self.nc.isel(x=slice(r.minx, r.maxx), y=slice(r.miny, r.maxy))

        # make sure the patch has the correct size
        assert nc_patch.sizes['x'] == 2 * self.patch_radius, \
            f'Wrong patch size: {nc_patch.sizes["x"]} != {2 * self.patch_radius}'
        assert nc_patch.sizes['y'] == 2 * self.patch_radius, \
            f'Wrong patch size: {nc_patch.sizes["y"]} != {2 * self.patch_radius}'

        data = self.process_data(ds=nc_patch, fp=self.fp)

        # save the patch location relative to the entire glacier (will be used later for mosaicking the predictions)
        data['patch_info'] = r.to_dict()

        return data

    def __len__(self):
        return len(self.patches_df)


class GlSegDataModule(pl.LightningDataModule):
    def __init__(self,
                 input_settings: dict,
                 split_csv: Path | str,
                 patches_on_disk: bool,
                 patches_dir: Path | str = None,
                 num_patches_train: int = None,
                 seed: int = 42,
                 cubes_dir: Path | str = None,
                 patch_radius: int = None,
                 stride_train: int = None,
                 stride_val: int = None,
                 stride_test: int = None,
                 standardize_data: bool = False,
                 minmax_scale_data: bool = False,
                 scale_each_band: bool = True,
                 norm_stats_csv: Path | str = None,
                 train_batch_size: int = 16,
                 val_batch_size: int = 32,
                 test_batch_size: int = 32,
                 train_shuffle: bool = True,
                 use_augmentation: bool = False,
                 num_workers: int = 16,
                 preload_data: bool = False,
                 pin_memory: bool = False):
        super().__init__()

        if not patches_on_disk:
            if patch_radius is None:
                raise ValueError('Patch radius must be provided when patches are not on disk')
            if stride_train is None:
                raise ValueError('Sampling step for training must be provided when patches are not on disk')
            if stride_val is None:
                raise ValueError('Sampling step for validation must be provided when patches are not on disk')
        else:
            if patches_dir is None:
                raise ValueError('Patches directory must be provided when patches are on disk')

        self.input_settings = input_settings
        self.patches_on_disk = patches_on_disk
        self.num_patches_train = num_patches_train
        self.seed = seed
        self.patch_radius = patch_radius
        self.stride_train = stride_train
        self.stride_val = stride_val
        self.stride_test = stride_test
        self.preload_data = preload_data
        self.standardize_data = standardize_data
        self.minmax_scale_data = minmax_scale_data
        self.scale_each_band = scale_each_band
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_shuffle = train_shuffle
        self.use_augmentation = use_augmentation
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # read the filepaths for all the patches, if provided, otherwise we will build them on the fly using the rasters
        data_dir = patches_dir if self.patches_on_disk else cubes_dir
        _label = 'Patches' if self.patches_on_disk else 'Rasters'
        if data_dir is None:
            raise ValueError(f'{_label} directory must be provided')
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f'{_label} directory {data_dir} does not exist')

        fp_list = sorted(list(Path(data_dir).rglob('*.nc')))
        log.info(f'Found {len(fp_list)} netcdf files in {data_dir}')

        # read the split and the corresponding train/val/test files
        log.info(f'Reading the split csv from {split_csv}')
        split_df = pd.read_csv(split_csv)

        # get the list of glaciers for each fold of the current split
        fp_list_per_fold = {}
        for fold_name in ['train', 'val', 'test']:
            glacier_ids = sorted(list(split_df[split_df['fold'] == fold_name].entry_id))
            fp_list_per_fold[fold_name] = sorted([fp for fp in fp_list if fp.parent.name in glacier_ids])

            # Make sure we have at least one file per glacier in the fold
            if len(fp_list_per_fold[fold_name]) == 0:
                raise FileNotFoundError(f"No files found for fold '{fold_name}' in {data_dir}")

        self.fp_list_train = fp_list_per_fold['train']
        self.fp_list_val = fp_list_per_fold['val']
        self.fp_list_test = fp_list_per_fold['test']

        # sanity checks
        assert len(set(self.fp_list_train) & set(self.fp_list_val)) == 0, 'Train and val overlap'
        assert len(set(self.fp_list_train) & set(self.fp_list_test)) == 0, 'Train and test overlap'
        assert len(set(self.fp_list_val) & set(self.fp_list_test)) == 0, 'val and test overlap'

        # check if some files were not assigned to any fold
        fp_list_assigned = set(self.fp_list_train) | set(self.fp_list_val) | set(self.fp_list_test)
        if len(fp_list_assigned) != len(fp_list):
            glacier_ids_missing = sorted([fp.parent.name for fp in set(fp_list) - fp_list_assigned])
            txt_missing = ', '.join(glacier_ids_missing[:10]) + ('...' if len(glacier_ids_missing) > 10 else '')
            log.warning(f'{len(glacier_ids_missing)} glaciers ({txt_missing}) were not assigned to any fold')

        # the following will be set when calling setup
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.test_ds_list = None

        # prepare the standardization constants if needed
        self.data_stats_df = None
        if self.standardize_data or self.minmax_scale_data:
            if norm_stats_csv is None:
                raise ValueError('Normalization stats CSV file must be provided when standardizing or scaling data')
            norm_stats_csv = Path(norm_stats_csv)
            if not norm_stats_csv.exists():
                raise FileNotFoundError(f'Normalization stats CSV file {norm_stats_csv} does not exist')
            log.info(f'Reading the normalization stats from {norm_stats_csv}')
            self.data_stats_df = pd.read_csv(norm_stats_csv)

    def setup(self, stage: str = None):
        if self.patches_on_disk:
            common_kwargs = dict(
                input_settings=self.input_settings,
                standardize_data=self.standardize_data,
                minmax_scale_data=self.minmax_scale_data,
                scale_each_band=self.scale_each_band,
                data_stats_df=self.data_stats_df
            )
            if stage == 'fit' or stage is None:
                self.train_ds = GlSegPatchDataset(
                    fp_list=self.fp_list_train,
                    use_augmentation=self.use_augmentation,
                    **common_kwargs,
                )
                self.val_ds = GlSegPatchDataset(fp_list=self.fp_list_val, **common_kwargs)
            elif stage == 'test':
                self.test_ds = GlSegPatchDataset(fp_list=self.fp_list_test, **common_kwargs)
        else:
            # build a dataset for each glacier (patches will be sampled on the fly), then concatenate them
            if stage == 'fit' or stage is None:
                self.train_ds = ConcatDataset(self.build_patch_dataset_per_glacier(
                    fp_rasters=self.fp_list_train,
                    use_augmentation=self.use_augmentation,
                    stride=self.stride_train
                ))

                # for validation, we don't apply any subsampling, adjust the sampling step if needed
                self.val_ds = ConcatDataset(self.build_patch_dataset_per_glacier(
                    fp_rasters=self.fp_list_val, stride=self.stride_val
                ))
            elif stage == 'test':
                # we enable add_extremes for the test set to make sure we have a good coverage of the glacier
                self.test_ds = ConcatDataset(self.build_patch_dataset_per_glacier(
                    fp_rasters=self.fp_list_test, stride=self.stride_test, add_extremes=True
                ))

        # Log the dataset sizes & subsample the training set if needed
        for ds_name, ds in zip(['train', 'val', 'test'], [self.train_ds, self.val_ds, self.test_ds]):
            if ds is None:
                continue
            if self.patches_on_disk:
                log.info(f"{ds_name} dataset: {len(ds)} patches from {ds.n_glaciers} glaciers")
            else:
                ds_sizes = [len(x) for x in ds.datasets]
                log.info(
                    f"{ds_name} dataset: "
                    f"{len(ds)} patches from {len(ds_sizes)} glaciers; "
                    f"#samples per glacier: min = {min(ds_sizes)} max = {max(ds_sizes)} avg = {np.mean(ds_sizes):.1f}"
                )

                if ds_name == "train" and self.num_patches_train is not None:
                    log.info(f"Subsampling the training set to {self.num_patches_train} patches")
                    self.subsample_train_ds(self.num_patches_train)
                    ds_sizes = [len(x) for x in self.train_ds.datasets]
                    log.info(
                        f"{ds_name} dataset: "
                        f"{len(self.train_ds)} patches from {len(ds_sizes)} glaciers; "
                        f"#samples per glacier: min = {min(ds_sizes)} max = {max(ds_sizes)} avg = {np.mean(ds_sizes):.1f}"
                    )

    def subsample_train_ds(self, n_patches):
        ds_list_train = self.train_ds.datasets
        ds_sizes_init = np.asarray([len(ds) for ds in ds_list_train])

        # Make sure we generated enough training patches (keep at least on patch per glacier)
        n_patches_init = sum(ds_sizes_init)
        if n_patches_init < n_patches:
            log.warning(
                f'Not enough patches for training: {n_patches_init} < {n_patches}. '
                f'Subsampling will not be performed.'
            )
            return

        n_glaciers = len(ds_list_train)

        # Keep at least one patch per glacier
        fraction = (n_patches - n_glaciers) / (n_patches_init - n_glaciers)
        ds_sizes_to_keep = np.asarray([1 + int(round((x - 1) * fraction)) for x in ds_sizes_init])

        # Make sure the sum is exactly the target number of patches
        idx = np.argsort(ds_sizes_to_keep)  # use the largest glaciers to make sure we have enough patches
        diff = n_patches - sum(ds_sizes_to_keep)
        ds_sizes_to_keep[idx[-abs(diff):]] += np.sign(diff)

        # Subsample
        for n, ds in zip(ds_sizes_to_keep, ds_list_train):
            ds.subsample(n=n, with_replacement=False, seed=self.seed)

        self.train_ds = ConcatDataset(ds_list_train)

    def build_patch_dataset_per_glacier(
            self, fp_rasters, use_augmentation=False, stride=None, add_extremes=False
    ):
        ds_list = run_in_parallel(
            fun=functools.partial(
                GlSegDataset,
                input_settings=self.input_settings,
                standardize_data=self.standardize_data,
                minmax_scale_data=self.minmax_scale_data,
                scale_each_band=self.scale_each_band,
                data_stats_df=self.data_stats_df,
                patch_radius=self.patch_radius,
                stride=stride,
                add_extremes=add_extremes,
                preload_data=self.preload_data,
                use_augmentation=use_augmentation
            ),
            fp=fp_rasters,
            pbar=True,
            pbar_desc='Preparing patch-level datasets for each glacier'
        )

        # make sure they're sorting by glacier ID (for reproducibility)
        ds_list = sorted(ds_list, key=lambda x: x.fp.parent.name)

        return ds_list

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False
        )

    def test_dataloaders_per_glacier(self, fp_rasters: list):
        """
        Build one dataloader per glacier (patches will be sampled on the fly) such that the predictions can be mosaicked
        together on epoch end.

        :param fp_rasters: list of filepaths to the rasters
        :return: list of dataloaders
        """
        test_ds_list = self.build_patch_dataset_per_glacier(
            fp_rasters=fp_rasters,
            stride=self.stride_test,
            add_extremes=True,
            use_augmentation=self.use_augmentation
        )

        dloaders = []
        for ds in test_ds_list:
            dloaders.append(
                DataLoader(
                    dataset=ds,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=False
                )
            )
        return dloaders
