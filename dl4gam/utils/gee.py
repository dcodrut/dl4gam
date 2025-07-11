import json
import logging
from pathlib import Path

import ee
import geedim as gd
import geedim.download
import geopandas as gpd
import pandas as pd
import shapely

from .parallel_utils import run_in_parallel

log = logging.getLogger(__name__)


def query_images(
        img_collection_name: str,
        geom: gpd.GeoSeries,
        start_date: str,
        end_date: str,
        gsd: int,
        bands: list = None,
        cloud_collection_name: str = 'COPERNICUS/S2_CLOUD_PROBABILITY',
        cloud_band: str = 'probability',
        cloud_mask_thresh_p: float = 0.4,
):
    """
    It queries the Google Earth Engine image collection for images that cover the specified region of interest (ROI)
    and adds the cloud mask band if a cloud collection is provided.
    If multiple images are available for the same acquisition date, it mosaics them.
    It also computes the coverage of the images over the ROI.

    :param img_collection_name: Google Earth Engine image collection name (e.g., 'COPERNICUS/S2_HARMONIZED')
    :param geom: the region of interest as a GeoSeries with a single geometry
    :param start_date: start date for filtering images (in 'YYYY-MM-dd' format)
    :param end_date: end date for filtering images (in 'YYYY-MM-dd' format)
    :param gsd: ground sample distance (pixel size in meters)
    :param bands: list of bands to keep in the images; if None, all bands are kept
    :param cloud_collection_name: Google Earth Engine image collection name for cloud masks (e.g., 'COPERNICUS/S2_CLOUD_PROBABILITY')
    :param cloud_band: which band to use for cloud probability (depends on the cloud collection)
    :param cloud_mask_thresh_p: which threshold (in [0, 1]) to use for binarizing the cloud probability
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
    if bands is not None:
        # First check if all the required bands are present in the images
        required_bands = set(bands)
        all_bands = set(ee.Image(imgs.first()).bandNames().getInfo())
        missing_bands = required_bands - all_bands
        if missing_bands:
            raise ValueError(
                f"The following required bands are missing from the collection: {', '.join(missing_bands)}. "
                f"Available bands: {', '.join(all_bands)}."
            )
        imgs = imgs.select(bands)

    ####################################################################################################################
    # Step 3: Reproject the images to the target CRS and scale
    ####################################################################################################################
    target_crs = ee.Projection(crs_orig)
    imgs = imgs.map(lambda img: ee.Image(img).resample('bilinear').reproject(crs=target_crs, scale=gsd))

    ####################################################################################################################
    # Step 4: Keep the latest processed tile per acquisition day (in case of multiple reprocessed versions)
    ####################################################################################################################
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
        # Get the band names and their scales
        img = ee.Image(img)
        bands = img.bandNames()
        band_scales = ee.Dictionary.fromLists(
            bands,
            bands.map(lambda b: ee.Number(img.select([b]).projection().nominalScale()))
        )

        # Get the band with the highest resolution (smallest scale)
        higres_band = ee.String(band_scales.keys().sort(band_scales.values()).get(0))
        mask = img.select([higres_band]).mask().rename('mask_ok')
        img = img.addBands(mask)
        return img

    def _compute_coverage(img):
        """
        Computes the coverage of an image over the region of interest.
        """
        img = ee.Image(img)
        coverage = img.select('mask_ok').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom_gee,
            scale=gsd,
            maxPixels=1e9
        ).get('mask_ok')

        return img.set({
            'coverage': coverage,
        })

    imgs = imgs.map(_add_data_coverage_mask)
    imgs = imgs.map(_compute_coverage)

    # From the full tiles, keep the first one alphabetically; for the partial tiles, keep all for later mosaicking.
    imgs_full = (
        imgs
        .filter(ee.Filter.eq('coverage', 1))
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
            Note that we include also missing pixels in the cloud mask, so that we can later sort only by cloud coverage.
            """
            cp_img = cloud_masks.filter(ee.Filter.eq('system:index', img.get('system:index'))).first()
            cp_img = ee.Image(cp_img).select('cloud_p')

            # If we are using the Google Cloud Score+ collection, invert & multiply the cloud probability by 100
            # (we will export the results as int16)
            if 'CLOUD_SCORE_PLUS' in cloud_collection_name:
                # 0 represents "not clear" (occluded), while 1 represents "clear" (unoccluded)
                # https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED#description
                cp_img = cp_img.multiply(-1).add(1).multiply(100).rename('cloud_p')

            # Save the cloud probability band to the image
            img = ee.Image(img).addBands(cp_img)

            # Binarize the probability map to create a binary mask & include the NODATA pixels
            cloud_mask = img.select('cloud_p').gt(cloud_mask_thresh_p * 100)
            mask_nok = img.select('mask_ok').eq(0)  # NODATA pixels
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

    # For the images with full coverage, save the properties as a dictionary (with a single element)
    imgs_full = imgs_full.map(
        lambda img: img.set({'metadata_tiles': ee.Dictionary.fromLists([img.get('id')], [img.toDictionary()])})
    )

    # Finally, we get one single image (or mosaic) per acquisition day, with all required bands (and cloud masks).
    imgs = imgs_full.merge(imgs_mosaics)

    return imgs


def prepare_geom(g: gpd.GeoSeries):
    """
    Validate and process the geometry for Google Earth Engine compatibility.
    :param g: gpd.GeoSeries - The geometry to validate
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

    # Return the geometry in the Earth Engine format
    return g.iloc[0].__geo_interface__


def compute_image_cloud_percentage(img: ee.Image, geom: gpd.GeoSeries, gsd: int = 10):
    """
    Compute cloud percentage for a single image.
    We expect the image to have a 'cloud_mask' band which is a binary mask where 1 indicates cloud presence.

    :param img: The image to process
    :param geom: Region to compute clouds over
    :param gsd: Ground sample distance (pixel size in meters)
    :return: ee.Image with cloud_p property added
    """

    # Validate and prepare the geometry
    geom_gee = prepare_geom(geom)

    avg_cloud_p = img.select('cloud_mask').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom_gee,
        scale=gsd,
        maxPixels=1e9
    ).get('cloud_mask')

    return img.set({'cloud_p': avg_cloud_p})


def compute_image_ndsi(img: ee.Image, geom: gpd.GeoSeries, bands_name_map: dict, gsd: int = 10):
    """
    Compute NDSI (Normalized Difference Snow Index) for a single image.

    :param img: yhe image to process
    :param geom: region to compute NDSI over
    :param bands_name_map: band name mapping dictionary (e.g., {'B3': 'G', 'B11': 'SWIR'})
    :param gsd: ground sample distance (pixel size in meters)
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

    cloud_free_mask = img.select('cloud_mask').eq(0)
    ndsi = img.normalizedDifference([bands_name_map_inv['G'], bands_name_map_inv['SWIR']]).rename('NDSI')
    ndsi_masked = ndsi.updateMask(cloud_free_mask)
    avg_ndsi = ndsi_masked.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom_gee,
        scale=gsd,
        maxPixels=1e9
    ).get('NDSI')

    return img.set({'ndsi': avg_ndsi})


def compute_image_albedo(img: ee.Image, geom: gpd.GeoSeries, bands_name_map: dict, gsd: int = 10):
    """
    Compute albedo for a single image.

    :param img: the image to process
    :param geom: region to compute albedo over
    :param bands_name_map: mapping of band names to their original names (e.g., {'B': 'B2', 'G': 'B3', 'R': 'B4'})
    :param gsd: ground sample distance (pixel size in meters)
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

    cloud_free_mask = img.select('cloud_mask').eq(0)
    albedo = img.expression(
        '0.5621 * B + 0.1479 * G + 0.2512 * R + 0.0015',
        {
            'R': img.select(bands_name_map_inv['R']),
            'G': img.select(bands_name_map_inv['G']),
            'B': img.select(bands_name_map_inv['B']),
        }
    ).rename('albedo')
    albedo_masked = albedo.updateMask(cloud_free_mask)
    avg_albedo = albedo_masked.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom_gee,
        scale=gsd,
        maxPixels=1e9
    ).get('albedo')

    return img.set({'albedo': avg_albedo})


def sort_images(df_meta: pd.DataFrame, sort_by: tuple, weights: tuple = None):
    """
    Computes a weighted score for each image based on the specified metrics and weights, then sorts the DataFrame by it.

    :param df_meta: DataFrame containing metadata for the images, including the metrics to sort by.
    :param sort_by: tuple of metric names to sort by (e.g., ('cloud_p', 'ndsi')).
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
        metadata: dict,
        geom: gpd.GeoSeries,
        gsd: int,
        skip_existing: bool = True,
        try_reading: bool = True,
        num_threads: int = None  # default is None, which uses the default from geedim.download.BaseImage.download
):
    """
    Downloads a single image from Google Earth Engine using the geedim library.

    :param fp: file path where the image will be saved (as a string or Path object).
    :param img: ee.Image object to download (can be a single image or a mosaic).
    :param metadata: dict containing metadata for the image, which will be saved as a JSON file alongside the image.
    :param geom: gpd.GeoSeries with a single geometry defining the region of interest (ROI) for downloading the image.
    :param gsd: ground sample distance (pixel size in meters) for the downloaded image.
    :param skip_existing: if True, skips downloading if the file already exists; if False, always downloads.
    :param try_reading: if True, attempts to read the existing file to check if it is valid before skipping download.
    :param num_threads: number of threads to use for downloading the image; if None, uses the default from geedim.download.BaseImage.download.
    :return: None
    """

    fp = Path(fp)

    # Convert the geometry for Earth Engine compatibility
    geom_gee = ee.Geometry(geom.to_crs('EPSG:4326').iloc[0].__geo_interface__)

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

    # Save the metadata to a JSON file
    with open(fp.with_suffix('.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


def download_best_images(
        img_collection_name: str,
        geom: gpd.GeoSeries,
        start_date: str,
        end_date: str,
        gsd: int,
        out_dir: str | Path,
        num_imgs_to_keep: int = 1,
        bands_name_map: dict = None,
        cloud_collection_name: str = 'COPERNICUS/S2_CLOUD_PROBABILITY',
        cloud_band: str = 'probability',
        cloud_mask_thresh_p: float = 0.4,
        max_cloud_p: float = 0.3,
        min_coverage: float = 0.9,
        sort_by: tuple = ('cloud_p', 'ndsi'),
        score_weights: tuple = None,
        geom_clouds: gpd.GeoSeries = None,
        geom_ndsi: gpd.GeoSeries = None,
        geom_albedo: gpd.GeoSeries = None,
        num_procs: int = 1,
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
    :param out_dir: directory where the images will be saved.
    :param num_imgs_to_keep: how many best images to keep and download.
    :param bands_name_map: dictionary re-mapping the band names (e.g., {'B3': 'G', 'B11': 'SWIR'}). We need this to compute NDSI and albedo, if requested.
    :param cloud_collection_name: collection name for cloud masks (e.g., 'COPERNICUS/S2_CLOUD_PROBABILITY').
    :param cloud_band: which band to use for cloud probability (depends on the cloud collection).
    :param cloud_mask_thresh_p: threshold for binarizing the cloud probability band (in [0, 1]).
    :param max_cloud_p: maximum allowed cloud percentage for the images to be considered (in [0, 1]), computed over the ROI.
    :param min_coverage: minimum required coverage of the images over the ROI (in [0, 1]).
    :param sort_by: tuple of metric names to sort by (e.g., ('cloud_p', 'ndsi')). The images will be sorted by these metrics using a weighted score.
    :param score_weights: weights for each metric in the sort_by tuple. If None, equal weights are used.
    :param geom_clouds: gpd.GeoSeries with a single geometry defining the region of interest for computing cloud percentage.
    :param geom_ndsi: gpd.GeoSeries with a single geometry defining the region of interest for computing NDSI.
    :param geom_albedo: gpd.GeoSeries with a single geometry defining the region of interest for computing albedo.
    :param num_procs: number of processes to use for downloading the images in parallel.
    :param skip_existing: if True, skips downloading if the file already exists; if False, always downloads.
    :param try_reading: if True, attempts to read the existing file to check if it is valid before skipping download.
    :return: None
    """

    log.info(f"Querying images from {img_collection_name} for {start_date} to {end_date}")
    imgs_all = query_images(
        img_collection_name=img_collection_name,
        geom=geom,
        start_date=start_date,
        end_date=end_date,
        gsd=gsd,
        bands=list(bands_name_map.keys()) if bands_name_map else None,
        cloud_collection_name=cloud_collection_name,
        cloud_band=cloud_band,
        cloud_mask_thresh_p=cloud_mask_thresh_p
    )

    # Compute the metrics for each image needed for sorting
    if 'cloud_p' in sort_by:
        imgs_all = imgs_all.map(lambda img: compute_image_cloud_percentage(ee.Image(img), geom_clouds, gsd))
    if 'ndsi' in sort_by:
        imgs_all = imgs_all.map(lambda img: compute_image_ndsi(ee.Image(img), geom_ndsi, bands_name_map, gsd))
    if 'albedo' in sort_by:
        imgs_all = imgs_all.map(lambda img: compute_image_albedo(ee.Image(img), geom_albedo, bands_name_map, gsd))

    # Save metadata to a DataFrame
    info = imgs_all.getInfo()
    feats = info['features']
    df_meta = pd.DataFrame([f['properties'] for f in feats])
    df_meta = df_meta.sort_values(by='date')

    # Print and export the statistics to a CSV file
    cols2keep = ['id', 'date', 'coverage'] + list(sort_by)
    df_meta_export = df_meta[cols2keep]
    with pd.option_context('display.max_columns', None, 'display.width', 240):
        fp_out_meta = Path(out_dir) / 'metadata_all.csv'
        fp_out_meta.parent.mkdir(parents=True, exist_ok=True)
        df_meta_export.to_csv(fp_out_meta, index=False)
        log.info(f"Stats (n = {len(df_meta_export)}) exported to {fp_out_meta}: \n{df_meta_export}")

    # Impose the minimum coverage and maximum cloud percentage
    df_meta = df_meta[(df_meta['coverage'] >= min_coverage) & (df_meta['cloud_p'] <= max_cloud_p)]
    log.info(
        f"After filtering by coverage >= {min_coverage} and cloud_p <= {max_cloud_p}, "
        f"we have {len(df_meta)} images left."
    )

    if len(df_meta) == 0:
        return  # No images left after filtering

    # Compute the QC scores, sort images and export the filtered metadata
    df_meta = sort_images(df_meta, sort_by, score_weights)
    cols2keep += [c for c in df_meta.columns if c.startswith('score')]
    df_meta_export = df_meta[cols2keep]
    with pd.option_context('display.max_columns', None, 'display.width', 240):
        fp_out_meta = Path(out_dir) / 'metadata_filtered.csv'
        df_meta_export.to_csv(fp_out_meta, index=False)
        log.info(f"Filtered stats (n = {len(df_meta_export)}) exported to {fp_out_meta}: \n{df_meta_export}")

    # Keep the best images based on the number requested
    if len(df_meta) < num_imgs_to_keep:
        log.warning(
            f"Requested {num_imgs_to_keep} images, but only {len(df_meta)} images left after filtering. "
            f"Downloading all available images."
        )
        num_imgs_to_keep = len(df_meta)

    # Download the best images in parallel
    ordered_ids = df_meta['id'].tolist()[:num_imgs_to_keep]
    fp_out_list = [Path(out_dir) / f"{img_id}.tif" for img_id in ordered_ids]
    metadata_list = [df_meta[df_meta['id'] == img_id].metadata_tiles.iloc[0] for img_id in ordered_ids]
    run_in_parallel(
        fun=download_image,
        img=[ee.Image(imgs_all.filter(ee.Filter.eq('id', img_id)).first()) for img_id in ordered_ids],
        fp=fp_out_list,
        metadata=metadata_list,
        geom=geom,
        gsd=gsd,
        skip_existing=skip_existing,
        try_reading=try_reading,
        num_procs=num_procs,
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
    _sort_by = ('cloud_p', 'ndsi')

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
        num_imgs_to_keep=4,
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
        num_procs=1,
        skip_existing=True,
        try_reading=True
    )
