import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

log = logging.getLogger(__name__)


def get_patches_df(
        nc: xr.Dataset,
        patch_radius: int,
        stride: int = None,
        add_center: bool = False,
        add_centroid: bool = False,
        add_extremes: bool = False,
        only_patches_centers_within_buffer: bool = True
):
    """
    Given a xarray dataset for one glacier, it returns a pandas dataframe with the pixel coordinates for square patches
    extracted from the dataset.

    :param nc: xarray dataset containing the image data and the glacier masks
    :param patch_radius: patch radius (in px)
    :param stride: sampling step applied both on x and y (in px);
        if smaller than 2 * patch_radius, then patches will overlap
    :param add_center: whether to add one patch centered in the middle of the glacier's box
    :param add_centroid: whether to add one patch centered in the centroid of the glacier
    :param add_extremes: whether to add four patches centered on the margin (in each direction) of the glacier
    :param only_patches_centers_within_buffer: whether to keep only the patches centered within the sampling mask
    :return: a pandas dataframe with the pixel coordinates of the patches
    """

    x_centers = []
    y_centers = []

    if stride is not None:
        if 'mask_patch_sampling' not in nc.data_vars:
            raise ValueError(
                f"Expected the netcdf file to contain a 'mask_patch_sampling' variable, but found {list(nc.data_vars)}. "
            )
        # Get all feasible patch centers s.t. the center pixel is on the provided sampling mask
        sampling_mask = (nc.mask_patch_sampling.data == 1)
        all_y_centers, all_x_centers = np.where(sampling_mask)

        # sample the feasible centers uniformly from either the glacier pixels or the whole image
        if only_patches_centers_within_buffer:
            idx_x = np.asarray([p % stride == 0 for p in all_x_centers])
            idx_y = np.asarray([p % stride == 0 for p in all_y_centers])
            idx = idx_x & idx_y
            x_centers = all_x_centers[idx]
            y_centers = all_y_centers[idx]
        else:
            x_centers = np.arange(0, nc.dims['x'], stride)
            y_centers = np.arange(0, nc.dims['y'], stride)
    else:
        if not (add_extremes or add_center or add_centroid):
            raise ValueError(
                'If stride is None, then at least one of add_extremes, add_center or add_centroid must be True.'
                'Otherwise, the function will not return any patches.'
            )

    # Get the glacier pixels
    if add_extremes or add_centroid or add_center:
        if 'mask_glacier' not in nc.data_vars:
            raise ValueError(
                f"Expected the netcdf file to contain a 'mask_glacier' variable, but found {list(nc.data_vars)}. "
            )

        glacier_mask = (nc.mask_glacier.data == 1)
        all_y_centers, all_x_centers = np.where(glacier_mask)
        minx = all_x_centers.min()
        miny = all_y_centers.min()
        maxx = all_x_centers.max()
        maxy = all_y_centers.max()

    if add_extremes:  # add the four patches centered on the margin of the glacier
        left = (minx, int(np.mean(all_y_centers[all_x_centers == minx])))
        top = (int(np.mean(all_x_centers[all_y_centers == miny])), miny)
        right = (maxx, int(np.mean(all_y_centers[all_x_centers == maxx])))
        bottom = (int(np.mean(all_x_centers[all_y_centers == maxy])), maxy)
        x_centers = np.concatenate([x_centers, [left[0], top[0], right[0], bottom[0]]])
        y_centers = np.concatenate([y_centers, [left[1], top[1], right[1], bottom[1]]])

    if add_centroid:
        x_centers = np.concatenate([x_centers, [int(np.mean(all_x_centers))]]).astype(int)
        y_centers = np.concatenate([y_centers, [int(np.mean(all_y_centers))]]).astype(int)

    if add_center:
        x_centers = np.concatenate([x_centers, [int((minx + maxx) / 2)]]).astype(int)
        y_centers = np.concatenate([y_centers, [int((miny + maxy) / 2)]]).astype(int)

    # build a geopandas dataframe with the sampled patches
    all_patches = {k: [] for k in ['x_center', 'y_center', 'minx', 'miny', 'maxx', 'maxy']}
    for x_center, y_center in zip(x_centers, y_centers):
        minx_patch, maxx_patch = x_center - patch_radius, x_center + patch_radius
        miny_patch, maxy_patch = y_center - patch_radius, y_center + patch_radius

        all_patches['x_center'].append(x_center)
        all_patches['y_center'].append(y_center)
        all_patches['minx'].append(minx_patch)
        all_patches['miny'].append(miny_patch)
        all_patches['maxx'].append(maxx_patch)
        all_patches['maxy'].append(maxy_patch)

    patches_df = pd.DataFrame(all_patches)

    return patches_df


def data_cv_split(
        geoms_fp: Path | str,
        num_folds: int,
        cv_iter: int,
        val_fraction: float,
        fp_out: Path | str
):
    """
    :param geoms_fp: path to the processed glacier outlines file
    :param num_folds: how many CV folds to use
    :param cv_iter: the current CV iteration
    :param val_fraction: the percentage of each training fold to be used as validation
    :param fp_out: where to save the csv with the split for the current CV iteration
    :return:
    """

    geoms_fp = Path(geoms_fp)
    if not geoms_fp.exists():
        raise FileNotFoundError(f'Geometries file {geoms_fp} does not exist.')

    if not 1 <= cv_iter <= num_folds:
        raise ValueError(f'cv_iter must be in the range [1, {num_folds}], but got {cv_iter}')

    # Read the outlines of the selected glaciers
    gl_df = gpd.read_file(geoms_fp, layer='glacier_sel')

    # Make sure there is a column with the area
    if 'area_km2' not in gl_df.columns:
        gl_df['area_km2'] = gl_df.geometry.area / 1e6

    # Regional split, assuming W to E direction (train-val-test)
    gl_df['bound_lim'] = gl_df.bounds.maxx
    gl_df = gl_df.sort_values('bound_lim', ascending=False)

    split_lims = np.linspace(0, 1, num_folds + 1)
    split_lims[-1] += 1e-4  # to include the last glacier

    # First extract the test fold and the combined train & val fold
    test_lims = (split_lims[cv_iter - 1], split_lims[cv_iter])
    area_cumsumf = gl_df.area_km2.cumsum() / gl_df.area_km2.sum()
    idx_test = (test_lims[0] <= area_cumsumf) & (area_cumsumf < test_lims[1])
    df_split = gl_df[['entry_id', 'area_km2', 'bound_lim']].copy()
    df_split.loc[idx_test, 'fold'] = 'test'

    # Choose the val set s.t. it acts as a clear boundary between test and train
    if cv_iter == 1:
        test_val_lims = (test_lims[0], test_lims[1] + val_fraction)
    elif cv_iter == num_folds:
        test_val_lims = (test_lims[0] - val_fraction, test_lims[1])
    else:
        test_val_lims = (test_lims[0] - val_fraction / 2, test_lims[1] + val_fraction / 2)
    idx_test_val = (test_val_lims[0] <= area_cumsumf) & (area_cumsumf < test_val_lims[1])
    idx_val = idx_test_val & (~idx_test)
    idx_train = ~idx_test_val
    df_split.loc[idx_val, 'fold'] = 'val'
    df_split.loc[idx_train, 'fold'] = 'train'

    # Make sure all glaciers are assigned to a fold
    if df_split.fold.isnull().any():
        raise ValueError(f'Some glaciers are not assigned to any fold.')

    # Log some statistics for the current split
    for fold in ['train', 'val', 'test']:
        df_fold = df_split[df_split.fold == fold]
        log.info(
            f'Fold {fold}: {(n := len(df_fold))} glaciers ({n / len(gl_df):.2%}); '
            f'area = {(s := df_fold.area_km2.sum()):.2f} km^2 ({s / gl_df.area_km2.sum():.2%})'
        )

    fp_out = Path(fp_out)
    fp_out.parent.mkdir(parents=True, exist_ok=True)
    df_split.to_csv(fp_out, index=False)
    log.info(f'Split for CV iteration {cv_iter} / {num_folds} saved to {fp_out}')


def patchify_data(
        cubes_dir: str | Path,
        patch_radius: int,
        patches_dir: str | Path,
        stride: int = None,
):
    """
    Runs the patchification process for all glaciers in the provided rasters directory.

    The patches will be later split into training, validation and test sets.
    When generating the patches, add_centroid will be set to True (see `get_patches_gdf`), which means at least one
    patche will be generated per glacier no matter the stride and patch radius.

    We sample patches uniformly from the `sampling_mask` variable in the netcdf files, which is expected to be
    a binary mask with 1s for the pixels where patches can be sampled and 0s otherwise. Additionally, we keep only the
    patches whose centers are falling within this mask (`only_patches_centers_within_buffer = True`).

    :param cubes_dir: directory containing the glacier-wide netcdf files
    :param patch_radius: patch radius (in px)
    :param stride: sampling step applied both on x and y (in px);
        if smaller than 2 * patch_radius, then patches will overlap.
    :param patches_dir: output directory where the extracted patches will be saved
    :return:
    """

    fp_list_all_g = sorted(list((Path(cubes_dir).rglob('*.nc'))))
    if len(fp_list_all_g) == 0:
        raise FileNotFoundError(f"No netcdf files found in {cubes_dir}. ")

    entry_id_list = sorted(set([fp.parent.name for fp in fp_list_all_g]))
    entry_id_to_fp = {x: [fp for fp in fp_list_all_g if fp.parent.name == x] for x in entry_id_list}

    if len(fp_list_all_g) > len(entry_id_list):
        # check which glaciers have more than one netcdf file
        print('The following glaciers have more than one netcdf file:')
        for entry_id, fp_list in entry_id_to_fp.items():
            if len(fp_list) > 1:
                print(f'{entry_id}: {len(fp_list)} files')
        raise ValueError(f"Expected one netcdf file per glacier in {cubes_dir}")

    for entry_id in tqdm(entry_id_list, desc='Patchifying'):
        g_fp = entry_id_to_fp[entry_id][0]
        nc = xr.open_dataset(g_fp, decode_coords='all', mask_and_scale=False).load()

        # get the locations of the sampled patches
        patches_df = get_patches_df(
            nc=nc,
            stride=stride,
            patch_radius=patch_radius,
            add_center=False,
            add_centroid=True,
            add_extremes=False,
            only_patches_centers_within_buffer=True
        )
        # build the patches
        for i in range(len(patches_df)):
            r = patches_df.iloc[0]
            nc_patch = nc.isel(x=slice(r.minx, r.maxx), y=slice(r.miny, r.maxy))

            fn = f'{entry_id}_patch_{i}_xc_{r.x_center}_yc_{r.y_center}.nc'

            patch_fp = Path(patches_dir) / entry_id / fn
            patch_fp.parent.mkdir(parents=True, exist_ok=True)

            nc_patch.to_netcdf(patch_fp)
