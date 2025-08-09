import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# local imports
from dl4gam.configs.datasets import QCMetric
from .data_prep import prep_glacier_dataset, extract_date_from_fn
from .gee import sort_images

log = logging.getLogger(__name__)


def compute_normalization_stats(fp):
    """
    Given the filepath to a data patch, it computes various statistics which will used to build the
    normalization constants (needed either for min-max scaling or standardization).

    :param fp: Filepath to a xarray dataset
    :return: a dictionary with the stats for the current raster
    """

    with xr.open_dataset(fp, decode_coords='all') as nc:
        band_data = nc.band_data.values
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


def compute_qc_stats(
        fp_img: str | Path,
        qc_metrics: tuple[QCMetric, ...],
        **kwargs,
):
    """
    In case images are already downloaded, this function computes the quality control statistics which can then be used
    for automated selection of the best images. If the images were downloaded using `dl4gam.workflow.download_gee_data`,
    then the statistics should be already computed.

    :param fp_img: Filepath to the image (tif) file.
    :param qc_metrics: The quality control metrics to compute.
        See `dl4gam.configs.datasets.QCMetric` for available metrics.
    :param kwargs: Additional keyword arguments passed to `prep_glacier_dataset`.
        Should include parameters like `bands_name_map`, `bands_nok_mask` and the ROI geometries for the QC metrics.
    :return: A dictionary with the computed statistics.
    """
    fp_img = Path(fp_img)
    stats = {
        'date': extract_date_from_fn(fp_img.name),
        'id': fp_img.stem
    }
    ds = prep_glacier_dataset(
        fp_img=fp_img,
        entry_id=fp_img.parent.name,
        **kwargs,
    )

    # Compute the coverage (count the number of non-NaN pixels in the band data)
    mask_na = (ds.band_data.values == ds.band_data.rio.nodata).any(axis=0)
    stats['coverage'] = 1 - np.sum(mask_na) / np.prod(mask_na.shape)

    # Compute the required QC metrics within the corresponding ROI
    # For the cloud percentage, we will use the NOK mask (see `bands_nok_mask` in the config)
    if QCMetric.CLOUD_P in qc_metrics:
        mask = (ds[f"mask_qc_roi_{QCMetric.CLOUD_P.value}"].values == 1)
        num_px_roi = np.sum(mask)
        mask_ok = (ds.mask_nok.values == 0) & ~mask_na
        num_px_ok_roi = np.sum(mask & mask_ok)
        stats[QCMetric.CLOUD_P.value] = 1 - (num_px_ok_roi / num_px_roi)

    # Exclude the NOK pixels before computing the other metrics
    ds = ds.where(ds.mask_nok.values == 0)
    if QCMetric.NDSI in qc_metrics:
        # We expect the G and SWIR bands to be present in the dataset
        required_bands = ['G', 'SWIR']
        if not all(b in ds.band_data.band.values for b in required_bands):
            raise ValueError(
                f"Cannot compute NDSI because the required bands {required_bands} are not present in the dataset."
                f"Make sure that the bands are present in the dataset and that the bands_name_map is set correctly."
            )
        # Compute the NDSI mask (G - SWIR) / (G + SWIR)
        mask = (ds[f"mask_qc_roi_{QCMetric.NDSI.value}"].values == 1)
        img_g_swir = ds.sel(band=['G', 'SWIR']).where(mask).band_data.values
        den = (img_g_swir[0] + img_g_swir[1])
        den[den == 0] = 1
        ndsi = (img_g_swir[0] - img_g_swir[1]) / den
        stats[QCMetric.NDSI.value] = np.nanmean(ndsi) if np.any(~np.isnan(ndsi)) else np.nan

    if QCMetric.ALBEDO in qc_metrics:
        # We expect the B, G, R bands to be present in the dataset
        required_bands = ['B', 'G', 'R']
        if not all(b in ds.band_data.band.values for b in required_bands):
            raise ValueError(
                f"Cannot compute albedo because the required bands {required_bands} are not present in the dataset."
                f"Make sure that the bands are present in the dataset and that the bands_name_map is set correctly."
            )
        # Compute the albedo (0.5621 * B + 0.1479 * G + 0.2512 * R + 0.0015)  # Wang et al., 2016
        mask = (ds[f"mask_qc_roi_{QCMetric.ALBEDO.value}"].values == 1)
        img_bgr = ds.sel(band=['B', 'G', 'R']).where(mask).band_data.values / 10000
        albedo = 0.5621 * img_bgr[0] + 0.1479 * img_bgr[1] + 0.2512 * img_bgr[2] + 0.0015
        stats[QCMetric.ALBEDO.value] = np.nanmean(albedo) if np.any(~np.isnan(albedo)) else np.nan

    # had some RAM issues, not sure why
    ds.close()  # close the dataset to avoid memory leaks
    del ds, mask_na
    gc.collect()  # force garbage collection

    return stats


def rank_images(
        raw_images_dir: str | Path,
        min_coverage: float = 0.9,
        max_cloud_p: float = 0.3,
        sort_by: tuple[QCMetric, ...] = (QCMetric.CLOUD_P, QCMetric.NDSI),
        score_weights: tuple = None,
        overwrite: bool = False,
        **kwargs,
):
    """
    Rank the images in the given (glacier) directory based on the quality control metrics.
    :param raw_images_dir: path to the directory containing the raw images (tif files).
    :param min_coverage: minimum coverage percentage to keep an image.
    :param max_cloud_p: maximum cloud percentage to keep an image.
    :param sort_by: metrics to sort the remaining images by.
        See `dl4gam.configs.datasets.QCMetric` for available metrics.
    :param score_weights: weights to use for the sorting metrics.
        If None, the weights will be set to 1 for all metrics in `sort_by`.
    :param overwrite: if True, overwrite the existing metadata file.
    :param kwargs: additional keyword arguments passed to `compute_qc_stats`.
    :return:
    """

    # First check if the metadata file already exists
    raw_images_dir = Path(raw_images_dir)
    fp_out_meta = raw_images_dir / 'metadata_all.csv'
    if fp_out_meta.exists() and not overwrite:
        log.info(f"Metadata file {fp_out_meta} exists and overwrite is set to False. Skipping computation.")
        df_meta = pd.read_csv(fp_out_meta)
    else:
        # Get all the (tif) images in the raw data directory
        fp_imgs = sorted(list(raw_images_dir.glob('*.tif')))
        if len(fp_imgs) == 0:
            log.warning(f"No images found in {raw_images_dir}. Please check the directory.")
            return

        # Compute the QC stats for each image
        stats_all = []
        for fp_img in fp_imgs:
            qc_metrics = list(sort_by)

            # Add the cloud percentage metric if not already present
            if QCMetric.CLOUD_P not in qc_metrics and max_cloud_p < 1:
                qc_metrics.append(QCMetric.CLOUD_P)

            stats = compute_qc_stats(fp_img, qc_metrics=qc_metrics, **kwargs)
            stats_all.append(stats)
        df_meta = pd.DataFrame(stats_all)

        # Save the metadata to a csv file
        fp_out_meta.parent.mkdir(parents=True, exist_ok=True)
        df_meta.to_csv(fp_out_meta, index=False)
        log.info(f"Metadata for {len(df_meta)} images exported to {fp_out_meta}")

    # Impose the minimum coverage and maximum cloud percentage
    if min_coverage > 0:
        df_meta = df_meta[df_meta['coverage'] >= min_coverage]

    if max_cloud_p < 1:
        df_meta = df_meta[df_meta[QCMetric.CLOUD_P] <= max_cloud_p]

    if len(df_meta) == 0:
        log.warning(
            f"No images left after filtering by coverage >= {min_coverage} and cloud_p <= {max_cloud_p}."
            f"raw_images_dir = {raw_images_dir}"
        )
        return

    # Sort the images
    df_meta = sort_images(df_meta, sort_by, score_weights)

    # Export the filtered metadata (drop unnecessary columns)
    cols2keep = ['id', 'date', 'coverage'] + [QCMetric.CLOUD_P, QCMetric.NDSI, QCMetric.ALBEDO]
    cols2keep += [c for c in df_meta.columns if c.startswith('score')]
    df_meta = df_meta[[c for c in cols2keep if c in df_meta.columns]]
    fp_out_meta = raw_images_dir / 'metadata_filtered.csv'
    df_meta.to_csv(fp_out_meta, index=False)

    with pd.option_context('display.max_columns', None, 'display.width', 300):
        log.info(f"Filtered metadata + QC metrics (n = {len(df_meta)}) exported to {fp_out_meta}: \n{df_meta}")
