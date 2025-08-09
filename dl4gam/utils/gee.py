import logging
from pathlib import Path
from typing import Optional

import ee
import geedim as gd
import geedim.download
import geopandas as gpd
import pandas as pd
import shapely

from dl4gam.configs.datasets import QCMetric
from dl4gam.utils.parallel_utils import run_in_parallel

log = logging.getLogger(__name__)


class EntryAdapter(logging.LoggerAdapter):
    """Logger adapter that adds entry_id context to all messages"""

    def process(self, msg, kwargs):
        return f"(entry_id = {self.extra['entry_id']}) {msg}", kwargs


def query_images(
        img_collection_name: str,
        geom: gpd.GeoSeries,
        start_date: str,
        end_date: str,
        bands: list | str = 'all',
        cloud_collection_name: Optional[str] = None,
        cloud_band: str = 'probability',
        cloud_mask_thresh_p: float = 0.4,
        latest_tile_only: bool = True
):
    """
    It queries the Google Earth Engine image collection for images that cover the specified region of interest (ROI)
    and adds the cloud mask band if a cloud collection is provided.
    If multiple images are available for the same acquisition date, it mosaics them.
    It also computes the coverage of the images over the ROI.

    :param img_collection_name: Google Earth Engine image collection name (e.g., 'COPERNICUS/S2_HARMONIZED')
    :param geom: the region of interest as a GeoSeries with a single geometry;
        Note than the data will be downloaded in the CRS of this geometry.
    :param start_date: start date for filtering images (in 'YYYY-MM-dd' format)
    :param end_date: end date for filtering images (in 'YYYY-MM-dd' format)
    :param bands: list of bands to keep in the images; if 'all', all bands are kept
    :param cloud_collection_name: Optional, Google Earth Engine image collection name for cloud masks
        (e.g., 'COPERNICUS/S2_CLOUD_PROBABILITY').
    :param cloud_band: which band to use for cloud probability (depends on the cloud collection)
    :param cloud_mask_thresh_p: which threshold (in [0, 1]) to use for binarizing the cloud probability
    :param latest_tile_only: if True, keeps only the last processed tile per acquisition day
        (in case of multiple reprocessed versions).
    :return: ee.ImageCollection with the filtered images
    """

    # Validate and prepare the geometry; save the original CRS before
    crs_orig = geom.crs.to_string()
    geom_gee = prepare_geom(geom)

    ####################################################################################################################
    # Step 1: Filter by date range and ROI bounds
    ####################################################################################################################
    imgs = ee.ImageCollection(img_collection_name).filterDate(start_date, end_date).filterBounds(geom_gee)

    # Save some properties which will need later
    imgs = imgs.map(lambda img: img.set({
        'id': img.get('system:index'),
        'processing_time': ee.String(img.get('system:index')).split('_').get(1),
        'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
        'tile': ee.String(img.get('system:index')).split('_').get(2)
    }))

    ####################################################################################################################
    # Step 2: Keep only the required bands if provided
    ####################################################################################################################
    if bands != 'all':
        # First check if all the required bands are present in the images
        required_bands = set(bands)
        all_bands = set(ee.Image(imgs.first()).bandNames().getInfo())
        missing_bands = required_bands - all_bands
        if missing_bands:
            raise ValueError(
                f"The following required bands are missing from the collection: {', '.join(missing_bands)}. "
                f"Available bands: {', '.join(all_bands)}."
            )
        imgs = imgs.select(list(bands))

    ####################################################################################################################
    # Step 3: Reproject the images to the target CRS
    ####################################################################################################################
    target_crs = ee.Projection(crs_orig)

    # Get the band with the highest resolution (smallest scale) to determine the GSD
    bands = imgs.first().bandNames()
    band_scales = bands.map(lambda b: ee.Number(imgs.first().select([b]).projection().nominalScale()))
    min_scale = band_scales.reduce(ee.Reducer.min())

    # Reproject the images to the target CRS using bilinear interpolation
    imgs = imgs.map(lambda img: ee.Image(img).resample('bilinear').reproject(crs=target_crs, scale=min_scale))

    ####################################################################################################################
    # Step 4: Keep the latest processed tile per acquisition day (in case of multiple reprocessed versions)
    ####################################################################################################################
    if latest_tile_only:
        imgs = imgs.map(
            lambda img: img.set({'date_tile_key': ee.String(img.get('date')).cat('_').cat(img.get('tile'))})
        )
        imgs = imgs.sort('processing_time', False).distinct('date_tile_key').sort('system:index', True)

    ####################################################################################################################
    # Step 5: Keep only one tile per acquisition day if already fully covers our ROI
    ####################################################################################################################
    def _add_data_coverage_mask(img):
        """
        Adds a mask band to the image indicating the coverage of the data over the region of interest.
        We use the band with the highest resolution (smallest scale) to create the mask.
        """
        img = ee.Image(img)

        # Get the band with the highest resolution (smallest scale)
        band_scales_dict = ee.Dictionary.fromLists(bands, band_scales)
        higres_band = ee.String(band_scales_dict.keys().sort(band_scales_dict.values()).get(0))
        mask = img.select(higres_band).mask().unmask(0).rename('mask_ok')
        img = img.addBands(mask)
        return img

    def _compute_coverage(img):
        """
        Computes the coverage of an image over the region of interest at the native resolution.
        """
        img = ee.Image(img)
        mask = img.select(['mask_ok'])

        # Count the number of pixels in the mask_ok band that are not NODATA
        sum_ones = mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom_gee,
            scale=min_scale,
            maxPixels=1e9
        ).get('mask_ok')
        area_ok = ee.Number(sum_ones).multiply(ee.Number(min_scale).pow(2))

        # Compute the total number of pixels in the region of interest
        area_target = ee.Geometry(geom_gee).area()
        coverage = area_ok.divide(area_target)

        return img.set({
            'coverage': coverage,
        })

    imgs = imgs.map(_add_data_coverage_mask)
    imgs = imgs.map(_compute_coverage)

    # From the full tiles, keep the first one alphabetically; for the partial tiles, keep all for later mosaicking.
    # We consider a tile to be fully covering the ROI if its coverage is >= 98% to allow for reprojection errors.
    imgs_full = (
        imgs
        .filter(ee.Filter.gte('coverage', 0.98))
        .sort('tile_code', True)
        .distinct('date')
    )

    # For the rest of the dates, we keep all tiles with partial coverage and mosaic them
    all_dates = imgs.aggregate_array('date').distinct()
    dates_full = imgs_full.aggregate_array('date').distinct()
    dates_needing_mosaics = ee.List(all_dates).removeAll(dates_full).sort()
    imgs_needing_mosaics = imgs.filter(ee.Filter.inList('date', dates_needing_mosaics))

    ####################################################################################################################
    # Step 6: Attach the cloud masks if a collection is provided
    ####################################################################################################################
    if cloud_collection_name is not None:
        cloud_masks = (
            ee.ImageCollection(cloud_collection_name)
            .select([cloud_band], ['cloud_p'])
            .filterDate(start_date, end_date)
            .filterBounds(geom_gee)
        )

        def _attach_cloud_mask(img):
            """
            Attaches the cloud probability band to the image from the cloud masks collection.
            Note that we also include the missing pixels in the cloud mask, so that later we can sort only by
            cloud coverage.
            """
            cp_img = cloud_masks.filter(ee.Filter.eq('system:index', img.get('system:index'))).first()
            cp_img = ee.Image(cp_img).select(['cloud_p'])

            # Reproject the cloud probability image to the same CRS as the image
            cp_img = cp_img.resample('bilinear').reproject(crs=target_crs, scale=min_scale)

            # If we are using the Google Cloud Score+ collection, invert & multiply the cloud probability by 100
            # (we will export the results as int16)
            if 'CLOUD_SCORE_PLUS' in cloud_collection_name:
                # 0 represents "not clear" (occluded), while 1 represents "clear" (unoccluded)
                # https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED#description
                cp_img = cp_img.multiply(-1).add(1).multiply(100).rename('cloud_p')

            # Save the cloud probability band to the image
            img = ee.Image(img).addBands(cp_img)

            # Binarize the probability map to create a binary mask & include the NODATA pixels
            cloud_mask = img.select(['cloud_p']).gt(cloud_mask_thresh_p * 100)
            mask_nok = img.select(['mask_ok']).eq(0)  # NODATA pixels
            cloud_mask = cloud_mask.Or(mask_nok)
            img = img.addBands(cloud_mask.rename('cloud_mask'))

            return img

        imgs_full = imgs_full.map(_attach_cloud_mask)
        imgs_needing_mosaics = imgs_needing_mosaics.map(_attach_cloud_mask)

    ####################################################################################################################
    # Step 7: Create mosaics for the dates with multiple tiles; save the metadata for all tiles in the mosaic
    ####################################################################################################################
    def _create_date_mosaic(date):
        """
        Creates a mosaic for a specific date from all tiles available for that date.
        Keep the metadata for all the tiles in the mosaic and recompute the coverage.
        """
        date_str = ee.String(date)
        imgs_crt_date = imgs_needing_mosaics.filter(ee.Filter.eq('date', date_str))
        mosaic = ee.ImageCollection(imgs_crt_date).mosaic()
        mosaic = mosaic.setDefaultProjection(target_crs)

        # recompute the coverage for the mosaic
        mosaic = _compute_coverage(mosaic)

        # set other properties for the mosaic
        return mosaic.set({
            'date': date_str,
            'id': imgs_crt_date.aggregate_array('system:index').join('-'),
            'tiles': imgs_crt_date.aggregate_array('tile'),
            'metadata_tiles': ee.Dictionary.fromLists(
                imgs_crt_date.aggregate_array('id'),
                imgs_crt_date.toList(imgs_crt_date.size()).map(lambda img: ee.Image(img).toDictionary())
            )
        })

    imgs_mosaics = ee.ImageCollection.fromImages(dates_needing_mosaics.map(_create_date_mosaic))

    # For the images with full coverage, save the properties/tiles as a dictionary/list (with a single element)
    imgs_full = imgs_full.map(
        lambda img: img.set({
            'tiles': ee.List([img.get('tile')]),
            'metadata_tiles': ee.Dictionary.fromLists([img.get('id')], [img.toDictionary()])
        })
    )

    # Finally, we get one single image (or mosaic) per acquisition day, with all required bands (and cloud masks).
    imgs = imgs_full.merge(imgs_mosaics)

    return imgs


def prepare_geom(g: gpd.GeoSeries, return_as_gee_geom: bool = True):
    """
    Validate and process the geometry for Google Earth Engine compatibility.
    :param g: gpd.GeoSeries - The geometry to validate
    :param return_as_gee_geom: bool - If True, coverts the geometry to a format compatible with Earth Engine.
    :return:
    """

    # Validate that the provided geometry is a GeoSeries with a single geometry.
    if not isinstance(g, gpd.GeoSeries):
        raise ValueError(f"Provided geometry must be a GeoSeries. Got {type(g)} instead.")

    if len(g) != 1:
        raise ValueError("Provided GeoSeries must contain a single geometry.")

    if g.crs is None:
        raise ValueError("Provided GeoSeries must have a valid CRS. Please set the CRS before using it.")

    # Project the geometry to WGS84 for Earth Engine compatibility
    g = g.to_crs('EPSG:4326')

    # Make sure the geometry is 2D (drop z-dimension if any)
    g = g.force_2d()

    if return_as_gee_geom:
        return g.iloc[0].__geo_interface__

    return g


def check_if_subset(geom_roi: gpd.GeoSeries, geom: gpd.GeoSeries, name: str):
    _geom_roi = prepare_geom(geom_roi, return_as_gee_geom=False)
    _geom = prepare_geom(geom, return_as_gee_geom=False)
    if not _geom.within(_geom_roi):
        raise ValueError(
            f"The provided geometry ({name}) is not a subset of the region of interest. "
            "Please ensure that the geometry is within the bounds of the ROI."
        )


def compute_image_cloud_percentage(img: ee.Image, geom: gpd.GeoSeries):
    """
    Compute cloud percentage for a single image at the native resolution.
    We expect the image to have a 'cloud_mask' band which is a binary mask where 1 indicates cloud presence.

    :param img: The image to process
    :param geom: Region to compute clouds over
    :return: ee.Image with cloud_p property added
    """

    # Validate and prepare the geometry
    geom_gee = prepare_geom(geom)

    cloud_mask = img.select(['cloud_mask'])
    avg_cloud_p = cloud_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom_gee,
        scale=cloud_mask.projection().nominalScale(),
        maxPixels=1e9
    ).get('cloud_mask')

    return img.set({QCMetric.CLOUD_P: avg_cloud_p})


def compute_image_ndsi(img: ee.Image, geom: gpd.GeoSeries, bands_name_map: dict):
    """
    Compute NDSI (Normalized Difference Snow Index) for a single image at its native resolution.

    :param img: yhe image to process
    :param geom: region to compute NDSI over
    :param bands_name_map: band name mapping dictionary (e.g., {'B3': 'G', 'B11': 'SWIR'})
    :return: ee.Image with ndsi property added
    """

    # Validate and prepare the geometry
    geom_gee = prepare_geom(geom)

    # Ensure that the map and the image have the required bands
    required_bands = ['G', 'SWIR']
    if not all(b in bands_name_map.values() for b in required_bands):
        raise ValueError(f"The bands_name_map must contain the following bands: {', '.join(required_bands)}.")

    # Invert the bands_name_map to get the mapping from band names to their original names
    bands_name_map_inv = {v: k for k, v in bands_name_map.items()}
    scale = img.select(bands_name_map_inv['G']).projection().nominalScale()
    cloud_free_mask = img.select(['cloud_mask']).eq(0)

    # First let's count the number of cloud-free pixels within the provided geometry
    num_cloud_free_px = cloud_free_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom_gee,
        scale=scale,
        maxPixels=1e9
    ).get('cloud_mask')

    # This will be computed if there are any cloud-free pixels
    ndsi = img.normalizedDifference([bands_name_map_inv['G'], bands_name_map_inv['SWIR']]).rename('NDSI')
    ndsi_masked = ndsi.updateMask(cloud_free_mask)
    avg_ndsi_masked = ndsi_masked.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom_gee,
        scale=scale,
        maxPixels=1e9
    ).get('NDSI')

    # If there are no cloud-free pixels, avg_ndsi makes no sense
    avg_ndsi = ee.Algorithms.If(
        ee.Number(num_cloud_free_px).gt(0),
        avg_ndsi_masked,
        ee.Number(1).erfInv()  # this will basically generate a string 'Infinity'; we will filter it out later
    )

    return img.set({QCMetric.NDSI: avg_ndsi})


def compute_image_albedo(img: ee.Image, geom: gpd.GeoSeries, bands_name_map: dict):
    """
    Compute albedo for a single image at its native resolution.

    :param img: the image to process
    :param geom: region to compute albedo over
    :param bands_name_map: mapping of band names to their original names (e.g., {'B': 'B2', 'G': 'B3', 'R': 'B4'})
    :return: ee.Image with albedo property added
    """

    # Validate and prepare the geometry
    geom_gee = prepare_geom(geom)

    # Ensure that the map and the image have the required bands
    required_bands = ['R', 'G', 'B']
    if not all(b in bands_name_map.values() for b in required_bands):
        raise ValueError(f"The bands_name_map must contain the following bands: {', '.join(required_bands)}.")

    # Invert the bands_name_map to get the mapping from band names to their original names
    bands_name_map_inv = {v: k for k, v in bands_name_map.items()}
    scale = img.select(bands_name_map_inv['R']).projection().nominalScale()
    cloud_free_mask = img.select(['cloud_mask']).eq(0)

    # First let's count the number of cloud-free pixels within the provided geometry
    num_cloud_free_px = cloud_free_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom_gee,
        scale=scale,
        maxPixels=1e9
    ).get('cloud_mask')

    # The following will be computed if there are any cloud-free pixels
    albedo = img.expression(
        '0.5621 * B + 0.1479 * G + 0.2512 * R + 0.0015',  # Wang et al., 2016
        {
            'R': img.select(bands_name_map_inv['R']),
            'G': img.select(bands_name_map_inv['G']),
            'B': img.select(bands_name_map_inv['B']),
        }
    ).rename('albedo')
    albedo_masked = albedo.updateMask(cloud_free_mask)
    avg_albedo_masked = albedo_masked.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom_gee,
        scale=scale,
        maxPixels=1e9
    ).get('albedo')

    # If there are no cloud-free pixels, avg_albedo makes no sense
    avg_albedo = ee.Algorithms.If(
        ee.Number(num_cloud_free_px).gt(0),
        avg_albedo_masked,
        ee.Number(1).erfInv()  # this will basically generate a string 'Infinity'; we will filter it out later
    )

    return img.set({QCMetric.ALBEDO: avg_albedo})


def sort_images(df_meta: pd.DataFrame, sort_by: tuple[QCMetric, ...], weights: tuple = None):
    """
    Computes a weighted score for each image based on the specified metrics and weights, then sorts the DataFrame by it.

    :param df_meta: DataFrame containing metadata for the images, including the metrics to sort by.
    :param sort_by: tuple of metric names to sort by (see QCMetric for available metrics).
    :param weights: weights for each metric in the sort_by tuple. If None, equal weights are used.
    :return: pd.DataFrame sorted by the computed score.
    """
    # We first scale each metric to [0, 1]
    for metric in sort_by:
        if metric not in df_meta.columns:
            raise ValueError(f"Metric '{metric}' not found in the metadata DataFrame.")
        den = df_meta[metric].max() - df_meta[metric].min()
        if den == 0:
            den = 1
        df_meta[f"score_{metric}"] = 1 - ((df_meta[metric] - df_meta[metric].min()) / den)
    # If weights are not provided, use equal weights
    if weights is None:
        weights = [1 / len(sort_by)] * len(sort_by)
    else:
        # make sure they sum up to 1
        weights = [x / sum(weights) for x in weights]

    # Compute the weighted score for each image and sort the DataFrame by it (lower is better)
    df_meta['score'] = sum(df_meta[f"score_{metric}"] * weight for metric, weight in zip(sort_by, weights))
    df_meta = df_meta.sort_values(by='score', ascending=False)

    return df_meta


def download_image(
        fp: str | Path,
        img: ee.Image,
        geom: gpd.GeoSeries,
        gsd: Optional[float] = None,
        skip_existing: bool = True,
        try_reading: bool = True,
        num_threads: int = None  # default is None, which uses the default from geedim.download.BaseImage.download
):
    """
    Downloads a single image from Google Earth Engine using the geedim library.

    :param fp: file path where the image will be saved (as a string or Path object).
    :param img: ee.Image object to download (can be a single image or a mosaic).
    :param geom: gpd.GeoSeries with a single geometry defining the region of interest (ROI) for downloading the image.
    :param gsd: ground sample distance (pixel size in meters) for the downloaded image.
        If None, GEE will decide the scale.
    :param skip_existing: if True, skips downloading if the file already exists; if False, always downloads.
    :param try_reading: if True, attempts to read the existing file to check if it is valid before skipping download.
    :param num_threads: number of threads to use for downloading the image;
        if None, uses the default from geedim.download.BaseImage.download.
    :return: None
    """

    fp = Path(fp)

    # Convert the geometry for Earth Engine compatibility
    geom_gee = prepare_geom(geom)

    # Check if the file already exists and skip downloading if requested
    if skip_existing and fp.exists():
        if not try_reading:
            log.info(f"File {fp} already exists. Skipping download.")
            return
        else:
            # try to read the file to increase the chance that it is valid
            try:
                import xarray as xr
                xr.open_dataarray(fp)
                log.info(f"File {fp} already exists and we could read it. Skipping download.")
                return
            except Exception as e:
                log.warning(f"File {fp} already exists but could not be read: {e}. Downloading again.")
                fp.unlink()  # remove the file to re-download it
                # remove the metadata also if it exists
                fp.with_suffix('.json').unlink(missing_ok=True)

    log.info(f"Downloading image to {fp}")

    # Convert the image to a geedim BaseImage object for downloading
    img = gd.download.BaseImage(img.clip(geom_gee))

    fp.parent.mkdir(parents=True, exist_ok=True)
    img.download(
        filename=fp,
        dtype='int16',
        overwrite=True,
        region=geom_gee,
        scale=gsd,
        num_threads=num_threads
    )


def download_best_images(
        img_collection_name: str,
        geom: gpd.GeoSeries,
        start_date: str,
        end_date: str,
        out_dir: str | Path,
        gsd: Optional[float] = None,
        num_days_to_keep: int = 1,
        latest_tile_only: bool = True,
        bands_to_keep: list | str = 'all',
        bands_name_map: Optional[dict] = None,
        cloud_collection_name: Optional[str] = None,
        cloud_band: str = 'probability',
        cloud_mask_thresh_p: float = 0.4,
        max_cloud_p: float = 1.0,
        min_coverage: float = 0.9,
        sort_by: Optional[tuple[QCMetric, ...]] = None,
        score_weights: Optional[tuple] = None,
        geom_clouds: Optional[gpd.GeoSeries] = None,
        geom_ndsi: Optional[gpd.GeoSeries] = None,
        geom_albedo: Optional[gpd.GeoSeries] = None,
        num_procs_download: int = 1,
        skip_existing: bool = True,
        try_reading: bool = True
):
    """
    Queries and downloads the best images from a Google Earth Engine image collection based on specified criteria.

    :param img_collection_name: Google Earth Engine image collection name (e.g., 'COPERNICUS/S2_HARMONIZED').
    :param geom: gpd.GeoSeries with a single geometry defining the region of interest (ROI) for downloading the images.
    :param start_date: start date for filtering images (in 'YYYY-MM-dd' format).
    :param end_date: end date for filtering images (in 'YYYY-MM-dd' format).
    :param gsd: ground sample distance (pixel size in meters) for the downloaded images.
        If None, we will let GEE decide. Note that this gsd will also be used for computing the various metrics.
    :param out_dir: directory where the images will be saved.
    :param num_days_to_keep: how many best images/days to keep and download.
    :param latest_tile_only: if True, keeps only the latest processed tile per acquisition day
        (in case of multiple reprocessed versions).
    :param bands_to_keep: list of bands to keep in the images; if 'all', all bands are kept.
    :param bands_name_map: dictionary re-mapping the band names (e.g., {'B3': 'G', 'B11': 'SWIR'}).
        We need this to compute NDSI and albedo, if requested.
    :param cloud_collection_name: collection name for cloud masks (e.g., 'COPERNICUS/S2_CLOUD_PROBABILITY') if cloud
    masks are needed (i.e. if geom_clouds is provided).
    :param cloud_band: which band to use for cloud probability (depends on the cloud collection).
    :param cloud_mask_thresh_p: threshold for binarizing the cloud probability band (in [0, 1]).
    :param max_cloud_p: maximum allowed cloud percentage for the images to be considered (in [0, 1]) computed over
        geom_clouds. Ignored if geom_clouds is None.
    :param min_coverage: minimum required coverage of the images over the ROI (in [0, 1]).
    :param sort_by: tuple of metric names to sort by (see QCMetric for available metrics).
        The images will be sorted by a score combining these metrics. If None, date is used for sorting.
    :param score_weights: weights for each metric in the sort_by tuple. If None, equal weights are used.
    :param geom_clouds: gpd.GeoSeries with a single geometry defining the ROI for computing cloud percentage.
        If None, we use the whole image ROI (geom).
    :param geom_ndsi: gpd.GeoSeries with a single geometry defining the ROI for computing NDSI.
        Note that only cloud-free pixels within this geometry will be used, so cloud mask collection is required.
        If None, we use the whole image ROI (geom).
    :param geom_albedo: gpd.GeoSeries with a single geometry defining the ROI for computing albedo.
        Note that only cloud-free pixels within this geometry will be used, so cloud mask collection is required.
        If None, we use the whole image ROI (geom).
    :param num_procs_download: number of processes to use for downloading the images in parallel.
    :param skip_existing: if True, skips downloading if the file already exists; if False, always downloads.
    :param try_reading: if True, attempts to read the existing file to check if it is valid before skipping download.
    :return: None
    """

    # Validate the geometries and other parameters
    _roi = prepare_geom(geom, return_as_gee_geom=False).iloc[0]

    # We will add the entry_id in the log messages
    _log = EntryAdapter(log, {'entry_id': geom.index[0]})

    # Compute the metrics for each image needed for filtering / sorting (if requested)
    metrics = ['coverage']  # always computed
    if sort_by is not None:
        for metric in sort_by:
            if metric not in QCMetric:
                msg = f"Invalid metric '{metric}' in sort_by. Available metrics: {', '.join(QCMetric)}."
                _log.error(msg)
                raise ValueError(msg)

        metrics += list(sort_by)

        # We always need to compute the cloud mask also for NDSI/albedo (only the cloud-free pixels are used)
        if QCMetric.CLOUD_P not in metrics:
            metrics.append(QCMetric.CLOUD_P)

    # We also need the cloud coverage if we want to filter by it
    if max_cloud_p < 1.0 and QCMetric.CLOUD_P not in metrics:
        metrics.append(QCMetric.CLOUD_P)

    # Now check if we need and have the cloud collection
    if QCMetric.CLOUD_P in metrics and cloud_collection_name is None:
        msg = "We need a valid cloud_collection_name. Not provided, but QCMetric.CLOUD_P is in required metrics."
        _log.error(msg)
        raise ValueError(msg)

    if geom_clouds is not None:
        geom_clouds = prepare_geom(geom_clouds, return_as_gee_geom=False)
        if not geom_clouds.iloc[0].within(_roi):
            msg = f"The geometry for computing cloud percentage must be within the main geometry."
            _log.error(msg)
            raise ValueError(msg)

        # We also expect a cloud collection to be provided if we want to compute cloud percentage
        if cloud_collection_name is None:
            msg = (
                "A cloud collection must be provided if you want to compute cloud percentage. "
                "Please provide a valid cloud_collection_name."
            )
            _log.error(msg)
            raise ValueError(msg)
    else:  # if no geometry for albedo is provided, we will use the whole image ROI
        geom_clouds = geom

    if geom_ndsi is not None:
        geom_ndsi = prepare_geom(geom_ndsi, return_as_gee_geom=False)
        if not geom_ndsi.iloc[0].within(_roi):
            msg = "The geometry for computing NDSI must be within the main geometry."
            _log.error(msg)
            raise ValueError(msg)
    else:  # if no geometry for albedo is provided, we will use the whole image ROI
        geom_ndsi = geom

    if geom_albedo is not None:
        geom_albedo = prepare_geom(geom_albedo, return_as_gee_geom=False)
        if not geom_albedo.iloc[0].within(_roi):
            msg = "The geometry for computing albedo must be within the main geometry."
            _log.error(msg)
            raise ValueError(msg)
    else:  # if no geometry for albedo is provided, we will use the whole image ROI
        geom_albedo = geom

    _log.info(f"Querying images from {img_collection_name} for {start_date} to {end_date}")
    imgs_all = query_images(
        img_collection_name=img_collection_name,
        geom=geom,
        start_date=start_date,
        end_date=end_date,
        bands=bands_to_keep,
        cloud_collection_name=cloud_collection_name,
        cloud_band=cloud_band,
        cloud_mask_thresh_p=cloud_mask_thresh_p,
        latest_tile_only=latest_tile_only
    )

    if QCMetric.CLOUD_P in metrics:
        imgs_all = imgs_all.map(lambda img: compute_image_cloud_percentage(ee.Image(img), geom_clouds))
    if QCMetric.NDSI in metrics:
        imgs_all = imgs_all.map(lambda img: compute_image_ndsi(ee.Image(img), geom_ndsi, bands_name_map))
    if QCMetric.ALBEDO in metrics:
        imgs_all = imgs_all.map(lambda img: compute_image_albedo(ee.Image(img), geom_albedo, bands_name_map))

    # Save metadata to a DataFrame
    info = imgs_all.getInfo()
    props = [f['properties'] for f in info['features']]

    if len(props) == 0:
        _log.warning("No images found with the specified criteria. Please check the input parameters.")
        return

    # Replace the `Infinity` values in the metrics with Python's float('inf')
    inf_val = ee.Number(1).erfInv().getInfo()
    for metric in metrics:
        for prop in props:
            if metric not in prop:
                raise ValueError(f"Metric '{metric}' not found in the image properties. ")

            if prop[metric] == inf_val:
                prop[metric] = float('inf')

    df_meta = pd.DataFrame(props)
    df_meta = df_meta.sort_values(by='date')
    _log.info(f"Found {len(df_meta)} images in the collection {img_collection_name} with the specified criteria.")

    # Drop the original metadata columns (we already have them in `metadata_tiles`)
    df_meta = df_meta[['date', 'processing_time', 'id', 'system:index', 'tiles'] + metrics + ['metadata_tiles']]

    # Print and export the statistics to a CSV file
    cols2show = ['id', 'date'] + metrics
    with pd.option_context('display.max_columns', None, 'display.width', 240):
        fp_out_meta = Path(out_dir) / 'metadata_all.csv'
        fp_out_meta.parent.mkdir(parents=True, exist_ok=True)
        df_meta.to_csv(fp_out_meta, index=False)
        _log.info(f"Stats (n = {len(df_meta)}) exported to {fp_out_meta}: \n{df_meta[cols2show]}")

    # Impose the minimum coverage and maximum cloud percentage
    if min_coverage > 0:
        df_meta = df_meta[df_meta['coverage'] >= min_coverage]
        _log.info(f"After filtering by coverage >= {min_coverage}, we have {len(df_meta)} images left.")

    if max_cloud_p < 1:
        df_meta = df_meta[df_meta[QCMetric.CLOUD_P] <= max_cloud_p]
        _log.info(f"After filtering by cloud_p <= {max_cloud_p}, we have {len(df_meta)} images left.")

    if len(df_meta) == 0:
        _log.warning("No images left after filtering. Please check the input parameters.")
        return

    # Compute the QC scores, sort images and export the filtered metadata
    if sort_by is not None:
        df_meta = sort_images(df_meta, sort_by, score_weights)

    # Export the filtered metadata with scores (drop the other columns)
    cols2keep = cols2show + [c for c in df_meta.columns if c.startswith('score')]
    df_meta_export = df_meta[cols2keep]
    with pd.option_context('display.max_columns', None, 'display.width', 240):
        fp_out_meta = Path(out_dir) / 'metadata_filtered.csv'
        df_meta_export.to_csv(fp_out_meta, index=False)
        _log.info(f"Filtered stats + scores (n = {len(df_meta_export)}) exported to {fp_out_meta}: \n{df_meta_export}")

    # Keep the best images based on the number requested
    if len(df_meta) < num_days_to_keep:
        _log.warning(
            f"Requested {num_days_to_keep} images, but only {len(df_meta)} images left after filtering. "
            f"Downloading all available images."
        )
        num_days_to_keep = len(df_meta)

    # Download the best images in parallel
    ordered_ids = df_meta['id'].tolist()[:num_days_to_keep]
    fp_out_list = [Path(out_dir) / f"{img_id}.tif" for img_id in ordered_ids]
    run_in_parallel(
        fun=download_image,
        img=[ee.Image(imgs_all.filter(ee.Filter.eq('id', img_id)).first()) for img_id in ordered_ids],
        fp=fp_out_list,
        geom=geom,
        gsd=gsd,
        skip_existing=skip_existing,
        try_reading=try_reading,
        num_procs=num_procs_download,
        pbar=(len(ordered_ids) > 1),  # show progress bar only if more than one image is downloaded
        num_threads=1  # this goes to geedim.download; use only one thread per image to avoid issues with GEE API limits
    )


if __name__ == "__main__":
    # Example usage

    # Initialize Earth Engine and prepare the ImageCollection
    ee.Initialize(project='your-project-id')

    # Setup console logging
    logging.basicConfig(level=logging.INFO)

    # To get a ROI as an example, we start with a location and create a ROI with 1 km buffer
    point = shapely.Point(9.249108, 46.912586)
    point_geoseries = gpd.GeoSeries([point], crs='EPSG:4326')
    _geom = point_geoseries.to_crs('EPSG:32632').buffer(2500)
    _geom_roi = _geom.buffer(2560).envelope  # rectangle buffered by 2560 m
    _gsd = 10  # Ground sample distance in meters

    # A map of band names to their original names in the collection
    # (it's needed later for identifying the bands required for NDSI and albedo)
    _bands_name_map = {
        'B1': 'B1',
        'B2': 'B',
        'B3': 'G',
        'B4': 'R',
        'B5': 'B5',
        'B6': 'B6',
        'B7': 'B7',
        'B8': 'NIR',
        'B8A': 'B8A',
        'B9': 'B9',
        'B10': 'B10',
        'B11': 'SWIR',
        'B12': 'B12'
    }

    # We will sort the images by cloud coverage & NDSI
    # (albedo can also be used, and you can also specify weights for each metric)
    _sort_by = (QCMetric.CLOUD_P, QCMetric.NDSI)

    # Let's assume we compute the cloud coverage on a 100m buffered geometry
    _geom_clouds = _geom.buffer(100)

    # Let's assume we compute the NDSI on a buffered geometry EXCLUDING the current geometry
    # (i.e. on non-glacier area which indicates seasonal snow cover)
    _geom_ndsi = _geom.buffer(100).difference(_geom)

    download_best_images(
        img_collection_name='COPERNICUS/S2_HARMONIZED',
        geom=_geom_roi,
        start_date='2015-08-01',
        end_date='2015-10-15',
        gsd=_gsd,
        out_dir=Path('/tmp/dl4gam_dl_example'),
        num_days_to_keep=4,
        bands_name_map=_bands_name_map,
        # cloud_collection_name='COPERNICUS/S2_CLOUD_PROBABILITY',
        # cloud_band='probability',
        # cloud_mask_thresh_p=0.4,
        cloud_collection_name='GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED',
        cloud_band='cs',
        cloud_mask_thresh_p=0.6,
        max_cloud_p=0.3,
        min_coverage=0.75,
        sort_by=_sort_by,
        score_weights=None,  # use equal weights for sorting
        geom_clouds=_geom_clouds,
        geom_ndsi=_geom_ndsi,
        num_procs_download=1,
        skip_existing=True,
        try_reading=True
    )
