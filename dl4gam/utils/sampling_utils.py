from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
import shapely.geometry
import xarray as xr
from tqdm import tqdm


def get_patches_gdf(nc, patch_radius, sampling_step=None, add_center=False, add_centroid=False, add_extremes=False):
    """
    Given a xarray dataset for one glacier, it returns a geopandas dataframe with the contours of square patches
    extracted from the dataset. The patches are generated only if they have the center pixel on the glacier.

    :param nc: xarray dataset containing the image data and the glacier masks
    :param patch_radius: patch radius (in px)
    :param sampling_step: sampling step applied both on x and y (in px);
                          if smaller than 2 * patch_radius, then patches will overlap
    :param add_center: whether to add one patch centered in the middle of the glacier's box
    :param add_centroid: whether to add one patch centered in the centroid of the glacier
    :param add_extremes: whether to add four patches centered on the margin (in each direction) of the glacier
    :return: a geopandas dataframe with the contours of the generated patches
    """

    if sampling_step is None:
        assert add_extremes or add_center or add_centroid, \
            'Enable at least one of add_extremes, add_center or add_centroid to generate at least one patch'

    # get all feasible patch centers s.t. the center pixel is on glacier
    assert 'mask_crt_g' in nc.data_vars, nc.data_vars
    nc_full_crt_g_mask_center_sel = (nc.mask_crt_g.data == 1)
    all_y_centers, all_x_centers = np.where(nc_full_crt_g_mask_center_sel)
    minx = all_x_centers.min()
    miny = all_y_centers.min()
    maxx = all_x_centers.max()
    maxy = all_y_centers.max()

    if sampling_step is not None:
        # sample the feasible centers uniformly; ensure that the first and last feasible centers are always included
        idx_x = np.asarray([p % sampling_step == 0 for p in all_x_centers])
        idx_y = np.asarray([p % sampling_step == 0 for p in all_y_centers])
        idx = idx_x & idx_y
        x_centers = all_x_centers[idx]
        y_centers = all_y_centers[idx]
    else:
        x_centers = []
        y_centers = []

    # add the four patches centered on the margin of the glacier
    if add_extremes:
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
    all_patches = {k: [] for k in ['x_center', 'y_center', 'bounds_px', 'bounds_m', 'gl_cvg']}
    for x_center, y_center in zip(x_centers, y_centers):
        minx_patch, maxx_patch = x_center - patch_radius, x_center + patch_radius
        miny_patch, maxy_patch = y_center - patch_radius, y_center + patch_radius
        nc_crt_patch = nc.isel(x=slice(minx_patch, maxx_patch), y=slice(miny_patch, maxy_patch))

        # compute the fraction of pixels that cover the glacier
        g_fraction = float(np.sum((nc_crt_patch.mask_crt_g == 1))) / nc_crt_patch.mask_crt_g.size

        all_patches['x_center'].append(x_center)
        all_patches['y_center'].append(y_center)
        all_patches['bounds_px'].append((minx_patch, miny_patch, maxx_patch, maxy_patch))
        all_patches['bounds_m'].append(nc_crt_patch.rio.bounds())
        all_patches['gl_cvg'].append(g_fraction)

    patches_df = gpd.GeoDataFrame(all_patches)
    patches_df = gpd.GeoDataFrame(patches_df, geometry=patches_df.bounds_m.apply(lambda x: shapely.geometry.box(*x)))
    patches_df = patches_df.set_crs(nc.rio.crs)

    return patches_df


def data_cv_split(gl_df, num_folds, valid_fraction, outlines_split_dir):
    """
    :param gl_df: geopandas dataframe with the glacier contours
    :param num_folds: how many CV folds to generate
    :param valid_fraction: the percentage of each training fold to be used as validation
    :param outlines_split_dir: output directory where the outlines split will be exported
    :return:
    """

    # make sure there is a column with the area
    assert 'area_km2' in gl_df.columns

    # regional split, assuming W to E direction (train-valid-test)
    gl_df['bound_lim'] = gl_df.bounds.maxx
    gl_df = gl_df.sort_values('bound_lim', ascending=False)

    split_lims = np.linspace(0, 1, num_folds + 1)
    split_lims[-1] += 1e-4  # to include the last glacier

    for i_split in range(num_folds):
        # first extract the test fold and the combined train & valid fold
        test_lims = (split_lims[i_split], split_lims[i_split + 1])
        area_cumsumf = gl_df.area_km2.cumsum() / gl_df.area_km2.sum()
        idx_test = (test_lims[0] <= area_cumsumf) & (area_cumsumf < test_lims[1])
        sdf_test = gl_df[idx_test]

        # choose the valid set s.t. it acts as a clear boundary between test and train
        if i_split == 0:
            test_valid_lims = (test_lims[0], test_lims[1] + valid_fraction)
        elif i_split == (num_folds - 1):
            test_valid_lims = (test_lims[0] - valid_fraction, test_lims[1])
        else:
            test_valid_lims = (test_lims[0] - valid_fraction / 2, test_lims[1] + valid_fraction / 2)
        idx_test_valid = (test_valid_lims[0] <= area_cumsumf) & (area_cumsumf < test_valid_lims[1])
        idx_valid = idx_test_valid & (~idx_test)
        idx_train = ~idx_test_valid
        sdf_train = gl_df[idx_train]
        sdf_valid = gl_df[idx_valid]

        df_per_fold = {
            'train': sdf_train,
            'valid': sdf_valid,
            'test': sdf_test,
        }

        for crt_fold, df_crt_fold in df_per_fold.items():
            print(f'Extracting outlines for split = {i_split + 1} / {num_folds}, fold = {crt_fold}')
            outlines_split_fp = Path(outlines_split_dir) / f'split_{i_split + 1}' / f'fold_{crt_fold}.shp'
            outlines_split_fp.parent.mkdir(parents=True, exist_ok=True)
            df_crt_fold.to_file(outlines_split_fp)
            print(
                f'Exported {len(df_crt_fold)} glaciers '
                f'out of {len(gl_df)} ({len(df_crt_fold) / len(gl_df) * 100:.2f}%);'
                f' actual area percentage = {df_crt_fold.area_km2.sum() / gl_df.area_km2.sum() * 100:.2f}%'
                f' ({df_crt_fold.area_km2.sum():.2f} km^2 from a total of {gl_df.area_km2.sum():.2f} km^2)')


def patchify_data(rasters_dir, patches_dir, patch_radius, sampling_step):
    """
    Using the get_patches_gdf function, it exports patches to disk for all glaciers.
    The patches will be later split into training, validation and test sets.
    When generating the patches, add_centroid and add_extremes will be set to True (see get_patches_gdf), which means
    at least five patches will be generated per glacier.

    :param rasters_dir: directory containing the raster netcdf files
    :param patches_dir: output directory where the extracted patches will be saved
    :param patch_radius: patch radius (in px)
    :param sampling_step: sampling step applied both on x and y (in px);
                          if smaller than 2 * patch_radius, then patches will overlap
    :return:
    """

    fp_list_all_g = sorted(list((Path(rasters_dir).rglob('*.nc'))))
    entry_id_list = sorted(set([fp.parent.name for fp in fp_list_all_g]))
    entry_id_to_fp = {x: [fp for fp in fp_list_all_g if fp.parent.name == x] for x in entry_id_list}

    if len(fp_list_all_g) > len(entry_id_list):
        # check which glaciers have more than one netcdf file
        print('The following glaciers have more than one netcdf file:')
        for entry_id, fp_list in entry_id_to_fp.items():
            if len(fp_list) > 1:
                print(f'{entry_id}: {len(fp_list)} files')
        raise ValueError(f"Expected one netcdf file per glacier in {rasters_dir}")

    for entry_id in tqdm(entry_id_list, desc='Patchifying'):
        g_fp = entry_id_to_fp[entry_id][0]
        nc = xr.open_dataset(g_fp, decode_coords='all', mask_and_scale=False).load()

        # get the locations of the sampled patches
        patches_df = get_patches_gdf(
            nc=nc,
            sampling_step=sampling_step,
            patch_radius=patch_radius,
            add_center=False,
            add_centroid=True,
            add_extremes=True
        )
        # build the patches
        for i in range(len(patches_df)):
            patch_shp = patches_df.iloc[i:i + 1]
            nc_patch = nc.rio.clip(patch_shp.geometry)

            r = patch_shp.iloc[0]
            fn = f'{entry_id}_patch_{i}_xc_{r.x_center}_yc_{r.y_center}.nc'

            patch_fp = Path(patches_dir) / entry_id / fn
            patch_fp.parent.mkdir(parents=True, exist_ok=True)

            nc_patch.to_netcdf(patch_fp)
