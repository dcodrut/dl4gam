import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# local imports
from .data_prep import prep_glacier_dataset


def compute_normalization_stats(fp):
    """
    Given the filepath to a data patch, it computes various statistics which will used to build the
    normalization constants (needed either for min-max scaling or standardization).

    :param fp: Filepath to a xarray dataset
    :return: a dictionary with the stats for the current raster
    """

    with xr.open_dataset(fp, decode_coords='all') as nc:
        band_data = nc.band_data.values[:13]  # TODO: parameterize the number of bands to consider
        list_arrays = [band_data]

        # add the other variables (except the masks and the already added band_data), assuming that they are 2D
        extra_vars = [v for v in nc if 'mask' not in v and v != 'band_data']
        for v in extra_vars:
            list_arrays.append(nc[v].values[None, ...])
        data = np.concatenate(list_arrays, axis=0)

        stats = {
            'entry_id': fp.parent.name,
            'fn': fp.name,
        }

        # add the stats for the band data
        n_list = []
        s_list = []
        ssq_list = []
        vmin_list = []
        vmax_list = []
        for i_band in range(len(data)):
            data_crt_band = data[i_band, :, :].flatten()
            all_na = np.all(np.isnan(data_crt_band))
            n_list.append(np.sum(~np.isnan(data_crt_band), axis=0) if not all_na else 0)
            s_list.append(np.nansum(data_crt_band, axis=0) if not all_na else np.nan)
            ssq_list.append(np.nansum(data_crt_band ** 2, axis=0) if not all_na else np.nan)
            vmin_list.append(np.nanmin(data_crt_band, axis=0) if not all_na else np.nan)
            vmax_list.append(np.nanmax(data_crt_band, axis=0) if not all_na else np.nan)

        stats['n'] = n_list
        stats['sum_1'] = s_list
        stats['sum_2'] = ssq_list
        stats['vmin'] = vmin_list
        stats['vmax'] = vmax_list
        stats['var_name'] = nc.band_data.long_name[:len(band_data)] + extra_vars

    return stats


def aggregate_normalization_stats(df):
    """
    Given the patch statistics computed using compute_normalization_stats, it combines them to estimate the
    normalization constants (needed either for min-max scaling or standardization).

    :param df: Pandas dataframe with the statistics for all the data patches.
    :return: a dataframe containing the normalization constants of each band.
    """

    # compute mean and standard deviation based only on the training folds
    stats_agg = {k: [] for k in ['var_name', 'mu', 'stddev', 'vmin', 'vmax']}
    for var_name in df.var_name.unique():
        df_r1_crt_var = df[df.var_name == var_name]
        n = max(df_r1_crt_var.n.sum(), 1)
        s1 = df_r1_crt_var.sum_2.sum()
        s2 = (df_r1_crt_var.sum_1.sum() ** 2) / n
        std = np.sqrt((s1 - s2) / n)
        mu = df_r1_crt_var.sum_1.sum() / n
        stats_agg['var_name'].append(var_name)
        stats_agg['mu'].append(mu)
        stats_agg['stddev'].append(std)
        stats_agg['vmin'].append(df_r1_crt_var.vmin.quantile(0.01))
        stats_agg['vmax'].append(df_r1_crt_var.vmax.quantile(0.99))
    df_stats_agg = pd.DataFrame(stats_agg)

    return df_stats_agg


def compute_qc_stats(gl_sdf, bands_name_map, bands_qc_mask, buffer_px):
    assert len(gl_sdf) == 1, 'Expecting a dataframe with a single entry.'
    row = gl_sdf.iloc[0]
    fp = Path(row.fp_img)
    stats = {'fp_img': str(fp), 'entry_id': row.entry_id}
    nc = prep_glacier_dataset(
        fp_img=fp,
        entry_id=row.entry_id,
        gl_df=gl_sdf,  # we need the mask only for the current glacier
        bands_name_map=bands_name_map,
        bands_qc_mask=bands_qc_mask,
        buffer_px=buffer_px,
        return_nc=True
    )

    # compute the fill percentage in the band data
    mask_na = (nc.band_data.values == nc.band_data.rio.nodata).any(axis=0)
    stats['fill_p'] = 1 - np.sum(mask_na) / np.prod(mask_na.shape)

    # prepare the QC mask (check the config to see what goes into the QC mask)
    mask_nok = (nc.mask_nok.data == 1) & ~mask_na

    # compute the cloud coverage stats over the entire scene, then over the glacier and, finally, over the glacier + 50m
    stats[f"cloud_p_scene"] = np.sum(mask_nok) / np.prod(mask_nok.shape)
    bg_mask = (nc.mask_crt_g.values == 1)
    assert np.sum(bg_mask) > 0, f'No glacier mask found in the image. row = {row.fp_img}'
    stats[f"cloud_p_gl"] = np.sum(mask_nok & bg_mask) / np.sum(bg_mask)
    bg_mask = (nc.mask_crt_g_b50.values == 1)
    assert np.sum(bg_mask) > 0, f'No glacier mask found in the image. row = {row}'
    stats[f"cloud_p_gl_b50m"] = np.sum(mask_nok & bg_mask) / np.sum(bg_mask)

    # compute the albedo, NDSI & Red \ SWIR, over the unclouded surfaces (glacier & non-glacier & within 50m buffer)
    mask_clean = ~mask_nok
    mask_gl_clean = (nc.mask_crt_g.values == 1) & mask_clean
    mask_gl_buffer_50m_clean = (nc.mask_crt_g_b50.values == 1) & mask_clean
    mask_non_gl_clean = (nc.mask_all_g_id.values == -1) & mask_clean
    mask_non_gl_clean_buffer_50m = (nc.mask_crt_g_b50.values == 1) & (nc.mask_all_g_id.values == -1) & mask_clean
    for name, mask in zip(
            ['scene', 'gl', 'gl_b50m', 'non_gl', 'non_gl_b50m'],
            [mask_clean, mask_gl_clean, mask_gl_buffer_50m_clean, mask_non_gl_clean, mask_non_gl_clean_buffer_50m]
    ):
        if np.sum(mask) >= 30:  # at least 30 clean pixels required
            nc_crt = nc.where(mask)

            # albedo
            img_bgr = nc_crt.sel(band=['B', 'G', 'R']).band_data.values / 10000
            albedo = 0.5621 * img_bgr[0] + 0.1479 * img_bgr[1] + 0.2512 * img_bgr[2] + 0.0015
            albedo_avg = np.nanmean(albedo) if np.sum(~np.isnan(albedo)) > 0 else np.nan

            # NDSI
            img_g_swir = nc_crt.sel(band=['G', 'SWIR']).band_data.values
            den = (img_g_swir[0] + img_g_swir[1])
            den[den == 0] = 1
            ndsi = (img_g_swir[0] - img_g_swir[1]) / den
            ndsi_avg = np.nanmean(ndsi) if np.sum(~np.isnan(ndsi)) > 0 else np.nan

            # Red / SWIR
            img_r_swir = nc_crt.sel(band=['R', 'SWIR']).band_data.values
            den = img_r_swir[1].copy()
            den[den == 0] = 1
            r_swir = img_r_swir[0] / den
            r_swir_avg = np.nanmean(r_swir) if np.sum(~np.isnan(r_swir)) > 0 else np.nan
        else:
            albedo_avg = np.nan
            ndsi_avg = np.nan
            r_swir_avg = np.nan

        stats[f"albedo_avg_{name}"] = albedo_avg
        stats[f"ndsi_avg_{name}"] = ndsi_avg
        stats[f"r_swir_avg_{name}"] = r_swir_avg

    # add also the statistics from the metadata if available
    fp_metadata = fp.with_suffix('.json')
    if fp_metadata.exists():
        with open(fp_metadata, 'r') as f:
            metadata = json.load(f)

        # get the cloud percentage for the entire image which geedim should automatically compute
        # if the image comes from multiple tiles, use the one with the highest coverage
        _fill_portion_list = [metadata['imgs_props'][k]['FILL_PORTION'] for k in metadata['imgs_props']]
        k = list(metadata['imgs_props'].keys())[np.argmax(_fill_portion_list)]
        cloudless_p_meta = metadata['imgs_props'][k]['CLOUDLESS_PORTION'] / 100
        fill_p_meta = metadata['imgs_props'][k]['FILL_PORTION'] / 100
        stats['cloud_p_meta'] = 1 - cloudless_p_meta
        stats['fill_p_meta'] = fill_p_meta

        # get the tile-level cloud percentage
        tile_level_cloud_p = metadata['imgs_props_extra'][k]['CLOUDY_PIXEL_PERCENTAGE'] / 100
        stats['tile_level_cloud_p'] = tile_level_cloud_p

    # had some RAM issues, not sure why
    nc.close()  # close the dataset to avoid memory leaks
    del nc, mask_na
    gc.collect()  # force garbage collection

    return stats
