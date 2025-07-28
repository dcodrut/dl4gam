import logging
import re
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio as rio
import rioxarray as rxr
import rioxarray.merge
import shapely
import shapely.ops
import xarray as xr
import xdem
from scipy.ndimage import gaussian_filter

log = logging.getLogger(__name__)

# regex for standalone YYYYMMDD
_DATE_REGEX = re.compile(r'(?<!\d)(\d{8})(?!\d)')


def extract_date_from_fn(fn: str) -> str | None:
    """
    Return the first YYYYMMDD found in the filename and returns the date as 'YYYY-MM-DD', or None if absent.
    """
    m = _DATE_REGEX.search(fn)
    if not m:
        return None

    # convert to 'YYYY-MM-DD'
    date_str = m.group(1)
    return pd.to_datetime(date_str, format='%Y%m%d').strftime('%Y-%m-%d')


def build_binary_mask(nc_data, geoms):
    """
    Build a binary mask for the given geometry using the given dataset.

    :param nc_data: the xarray dataset with the (raw) image data
    :param geoms: the list of geometries to be used for building the mask
    :return: the binary mask
    """

    # Get a single band
    tmp_raster = nc_data.band_data.isel(band=0)

    # Ensure the nodata value is NaN (should be the case if xr.open_dataset was called with mask_and_scale=True)
    if not np.isnan(tmp_raster.rio.nodata):
        tmp_raster = xr.decode_cf(tmp_raster.to_dataset(name='data'))['data']
        assert np.isnan(tmp_raster.rio.nodata), \
            f"Unexpected error: the nodata value is not NaN, but {tmp_raster.rio.nodata}."

    # Clip the raster using the given geometries and get the binary mask
    mask = tmp_raster.rio.clip(geoms, drop=False).notnull().values

    return mask


def crop_and_add_glacier_masks(
        nc_data,
        gl_df,
        entry_id_int,
        buffer=0,
        check_data_coverage=True,
        buffers_masks=(0,)
):
    """
    Add glacier masks to the given dataset after cropping it using the given buffer.

    The masks are:
        * mask_all_glaciers_id: a mask with the IDs of all the glaciers in the current glacier's bounding box
        * mask_glacier: a binary mask with the current glacier
        * mask_glacier_bX: a binary mask with the current glacier and a buffer of X meters around it

    :param nc_data: the xarray dataset with the (raw) image data
    :param gl_df: the geopandas dataframe with all the glaciers' outlines
    :param entry_id_int: the integer ID of the current glacier
    :param buffer: the buffer around the current glacier (in meters); it is used to crop the image data
    :param check_data_coverage: whether to check if the data covers the current glacier + buffer
    :param buffers_masks: the buffers around the current glacier (in meters) for which binary masks are created

    :return: the (cropped) dataset with the added masks
    """

    # project the glaciers' outlines on the same CRS as the given dataset
    gl_proj_df = gl_df.to_crs(nc_data.rio.crs)

    # get the outline of the current glacier
    crt_g_shp = gl_proj_df[gl_proj_df.entry_id_i == entry_id_int]

    # create a bounding box which contains any possible patch which will be sampled around the current glacier
    g_buff_bbox = crt_g_shp.geometry.iloc[0].envelope.buffer(buffer).envelope

    # check if the bounding box with the buffer is completely contained in the data boundaries
    data_bbox = shapely.geometry.box(*nc_data.rio.bounds())
    assert not check_data_coverage or data_bbox.contains(g_buff_bbox), \
        f"The data does not include the current glacier {crt_g_shp.entry_id.iloc[0]} and a {buffer}m buffer"

    # keep only the current glacier and its neighbours
    nc_data_crop = nc_data.rio.clip([g_buff_bbox])

    # pad the data with NODATA values if the glacier is not fully covered
    if not check_data_coverage:
        nc_data_crop = nc_data_crop.rio.pad_box(*g_buff_bbox.bounds)

    # add masks (current glacier mask & all glaciers mask, i.e. with glaciers IDs)
    # 1. all glaciers mask - with glaciers IDs
    mask_rgi_id = np.zeros_like(nc_data_crop.band_data.values[0], dtype=np.int32) - 1
    gl_crt_buff_df = gl_proj_df[gl_proj_df.intersects(g_buff_bbox)]
    for i in range(len(gl_crt_buff_df)):
        row = gl_crt_buff_df.iloc[i]
        mask_crt_g = build_binary_mask(nc_data_crop, geoms=[row.geometry])
        mask_rgi_id[mask_crt_g] = row.entry_id_i
    nc_data_crop['mask_all_glaciers_id'] = (('y', 'x'), mask_rgi_id)
    nc_data_crop['mask_all_glaciers_id'].attrs['_FillValue'] = -1

    # 2. binary mask only for the current glacier, also with various buffers (in meters)
    for buffer_mask in buffers_masks:
        _crt_g_shp = crt_g_shp.buffer(buffer_mask).iloc[0]
        if not _crt_g_shp.is_empty:
            mask_crt_g = build_binary_mask(nc_data_crop, geoms=[_crt_g_shp])
        else:
            mask_crt_g = np.zeros_like(nc_data_crop.mask_all_glaciers_id)

        label = '' if buffer_mask == 0 else f'_b{buffer_mask}'
        k = 'mask_glacier' + label
        nc_data_crop[k] = (('y', 'x'), mask_crt_g.astype(np.int8))
        nc_data_crop[k].attrs['_FillValue'] = -1

    return nc_data_crop


def prep_glacier_dataset(
        fp_img: str | Path,
        entry_id: str,
        gl_df: gpd.GeoDataFrame,
        buffer: int = 0,
        check_data_coverage=True,
        bands_name_map: Optional[dict[str, str]] = None,
        bands_nok_mask: Optional[list[str]] = None,
        extra_geodataframes: Optional[dict[str, gpd.GeoDataFrame]] = None,
        extra_rasters: Optional[dict[str, str | Path]] = None,
        xdem_features: Optional[list[str]] = None,
        no_data: int = -9999,
        fp_out: Optional[str | Path] = None,
        overwrite: bool = False
):
    """
    Prepare a glacier dataset by cropping the raw data within a buffer around the glacier outline, and then
    adding the glacier masks (one for the current glacier, and one for all glaciers in the bounding box).
    Optionally, it can also add extra vector data (e.g. debris cover) and extra raster data (e.g. DEM features).

    :param fp_img: the path to the raw image
    :param entry_id: the ID of the current glacier
    :param gl_df: the geopandas dataframe with all the glacier outlines
    :param buffer: the buffer (in meters) around the current glacier to be used when cropping the image data
    :param check_data_coverage: whether to check if the data covers the current glacier + buffer
    :param bands_name_map: A dict containing the bands we keep from the raw data (as keys) and their new names (values);
        If None, all the bands are kept, with their original names.
    :param bands_nok_mask: the names of the bands to be used for building the mask for bad pixels
    :param extra_geodataframes: a dict with keys as variable names and values as GeoDataFrames containing the extra
        vector data (e.g. debris cover) to be added as additional binary masks
    :param extra_rasters: a dict with keys as variable names and values as directories containing the rasters (.tif) to
        be added as additional variables
    :param xdem_features: a list of DEM features to be added to the dataset computed with xDEM.
        (see https://xdem.readthedocs.io/en/stable/gen_modules/xdem.DEM.get_terrain_attribute.html).
        Any attribute can be used, e.g. ['slope', 'aspect', 'profile_curvature', 'terrain_ruggedness_index'].
        If None, no DEM features are added.
    :param no_data: the value to be used as NODATA for the raster data; for the binary masks, it will be -1
    :param fp_out: the path to the output glacier dataset (if None, the raster is returned)
    :param overwrite: whether to overwrite the output file if it already exists
    :return: None or the xarray dataset
    """
    row_crt_g = gl_df[gl_df.entry_id == entry_id]
    assert len(row_crt_g) == 1

    # Read the raw image
    log.debug(f"Reading the raw image {fp_img} for glacier {entry_id}")
    ds = xr.open_dataset(fp_img, mask_and_scale=False)

    # Set the NODATA value for the dataset and the band data
    ds.attrs['_FillValue'] = no_data
    ds.band_data.attrs['_FillValue'] = no_data

    # Save the glacier ID in the dataset attributes
    ds.attrs['entry_id'] = entry_id

    # Check if the name of the bands is given, otherwise name them
    if 'long_name' not in ds.band_data.attrs:
        ds.band_data.attrs['long_name'] = [f'B{i + 1}' for i in range(len(ds.band_data))]

    # Set the band names for indexing
    ds = ds.assign_coords(band=list(ds.band_data.long_name))

    # Build the mask for the bad (e.g. cloudy, shadowed or missing) pixels if needed
    if bands_nok_mask is not None:
        # Ensure the bands to keep are in the image
        bands_missing = [b for b in bands_nok_mask if b.replace('~', '') not in ds.band_data.long_name]
        assert len(bands_missing) == 0, f"{bands_missing} not found in the image bands = {ds.band_data.long_name}"

        # Build the mask
        mask_nok = np.zeros_like(ds.band_data.isel(band=0).values, dtype=bool)
        for b in bands_nok_mask:
            crt_mask = (ds.band_data.sel(band=b.replace('~', '')).values == 1)
            # Invert the mask if needed
            if b[0] == '~':
                crt_mask = ~crt_mask
            mask_nok |= crt_mask

        # Add the mask to the dataset
        ds['mask_nok'] = (('y', 'x'), mask_nok.astype(np.int8))
        ds['mask_nok'].attrs['_FillValue'] = -1
        ds['mask_nok'].attrs['bands_mask_nok'] = tuple(bands_nok_mask)

    # Keep only the bands we will need later if given
    if bands_name_map is not None:
        # Ensure the bands to keep are in the image
        bands_missing = [b for b in bands_name_map if b not in ds.band_data.long_name]
        assert len(bands_missing) == 0, f"{bands_missing} not found in the image bands = {ds.band_data.long_name}"

        # Subset the bands
        bands_to_keep = list(bands_name_map.keys())
        ds = ds.sel(band=list(bands_to_keep))

        # Rename the bands
        new_band_names = [bands_name_map[b] for b in bands_to_keep]
        ds = ds.assign_coords(band=new_band_names)
        ds.band_data.attrs['long_name'] = new_band_names
        ds.band_data.attrs['description'] = new_band_names

    # Add the glacier masks
    log.debug(f"Cropping and adding glacier masks for glacier {entry_id} with buffer {buffer}m")
    entry_id_int = row_crt_g.iloc[0].entry_id_i
    ds = crop_and_add_glacier_masks(
        nc_data=ds,
        gl_df=gl_df,
        entry_id_int=entry_id_int,
        buffer=buffer,
        check_data_coverage=check_data_coverage
    )

    # Add the extra vector data if provided
    if extra_geodataframes is not None:
        log.debug(f"Adding extra masks (names = {list(extra_geodataframes.keys())}) for glacier {entry_id}")
        ds = add_extra_vectors(ds, extra_geodataframes)

    # Add the extra raster data if provided
    if extra_rasters is not None:
        log.debug(f"Adding extra rasters (names = {list(extra_rasters.keys())}) for glacier {entry_id}")
        ds = add_extra_rasters(ds, extra_rasters, no_data)

    # Add DEM features computed with xDEM
    if xdem_features is not None:
        if 'dem' in ds.data_vars:
            log.debug(f"Adding DEM features ({xdem_features}) for glacier {entry_id}")
            ds = add_dem_features(ds, xdem_features, no_data)
        else:
            raise ValueError(
                f"The dataset for glacier {entry_id} does not contain a 'dem' variable, "
                f"so we cannot compute DEM features. "
            )

    # Not sure why but needed for QGIS
    for v in ds.data_vars:
        ds[v].rio.write_crs(ds.rio.crs, inplace=True)

    # export if needed
    if fp_out is not None:
        fp_out.parent.mkdir(exist_ok=True, parents=True)
        ds.attrs['fn'] = fp_img.name
        ds.attrs['glacier_area'] = row_crt_g.area_km2.iloc[0]

        # Delete the existing file if needed
        if fp_out.exists():
            if overwrite:
                log.warning(f"Deleting the existing file {fp_out}")
                fp_out.unlink()
            else:
                raise FileExistsError(
                    f"The file {fp_out} already exists. Delete it or use overwrite=True to overwrite it."
                )

        ds.to_netcdf(fp_out)
        ds.close()
        return None
    else:
        return ds


def add_extra_vectors(ds: xr.Dataset, extra_geodataframes: dict[str, gpd.GeoDataFrame]):
    """
    Add extra vector data to the glacier dataset, e.g. debris cover, as additional variables (binary masks).

    :param ds: the Xarray glacier dataset to which the extra vectors will be added.
    :param extra_geodataframes: dict with keys as variable names and values as GeoDataFrames containing the extra vector data (e.g. debris cover).

    :return: the updated dataset
    """

    for k, gdf in extra_geodataframes.items():
        gdf_proj = gdf.to_crs(ds.rio.crs)
        mask = build_binary_mask(ds, geoms=gdf_proj.geometry.values)
        mask_name = f"mask_{k}"
        ds[mask_name] = (('y', 'x'), mask.astype(np.int8))
        ds[mask_name].attrs['_FillValue'] = -1

    return ds


def add_extra_rasters(ds: xr.Dataset, extra_rasters: dict[str, str | Path], no_data: int):
    """
    Add extra rasters to the glacier dataset as additional variables.

    For the given glacier dataset, it checks which of the provided rasters intersect the glacier's bounding box,
    reprojects them to the glacier's CRS, merges them if needed, and adds them as new variables.

    :param ds: the Xarray glacier dataset to which the extra rasters will be added.
    :param extra_rasters: dict with keys as variable names and values as directories containing the rasters (.tif).
    :param no_data: the value to be used as NODATA

    :return: the updated dataset
    """

    # Save the bounding boxes of all the provided rasters (needed for computing the intersection with the glacier)
    extra_rasters_bb_dict = {}
    for k, crt_dir in extra_rasters.items():
        rasters_crt_dir = list(Path(crt_dir).rglob('*.tif'))
        extra_rasters_bb_dict[k] = {
            fp: shapely.geometry.box(*xr.open_dataset(fp).rio.bounds()) for fp in rasters_crt_dir
        }

    # Save the bounding box of the current glacier dataset and add a buffer to make sure we include all the rasters
    # possibly intersecting the current glacier (otherwise we may miss some pixels due to re-projection errors)
    ds_bbox = shapely.geometry.box(*ds.rio.bounds())
    ds_bbox = ds_bbox.buffer(100)

    for k, crt_extra_rasters_bb_dict in extra_rasters_bb_dict.items():
        # Check which raster files intersect the current glacier
        crt_nc_list = []
        for fp, raster_bbox in crt_extra_rasters_bb_dict.items():
            crt_nc = xr.open_dataarray(fp, mask_and_scale=False)
            transform = pyproj.Transformer.from_crs(crt_nc.rio.crs, ds.rio.crs, always_xy=True).transform
            if ds_bbox.intersects(shapely.ops.transform(transform, raster_bbox)):
                crt_nc_list.append(crt_nc)

        # Ensure at least one intersection was found
        assert len(crt_nc_list) > 0, f"No intersection found for {k} (glacier ID = {ds.attrs['entry_id']})"

        # Merge the datasets if needed
        raster = rxr.merge.merge_arrays(crt_nc_list) if len(crt_nc_list) > 1 else crt_nc_list[0]

        # Reproject
        raster = (
            raster.isel(band=0).rio.reproject_match(ds, resampling=rasterio.enums.Resampling.bilinear)
            .astype(np.float32)
        )

        # Set the NODATA value
        raster.attrs['_FillValue'] = np.float32(no_data)

        # Add the current raster to the glacier dataset
        ds[k] = raster

    return ds


def add_dem_features(ds: xr.Dataset, xdem_features: list[str], no_data: int = -9999):
    """
    Add DEM features to the glacier dataset using the XDEM library.
    The features are: slope, aspect, planform curvature, profile curvature, terrain ruggedness index.

    :param ds: the Xarray glacier dataset to which the DEM features will be added.
    :param xdem_features: a list of DEM features to be added to the dataset computed with xDEM.
        See https://xdem.readthedocs.io/en/stable/gen_modules/xdem.DEM.get_terrain_attribute.html
    :param no_data: the value to be used as NODATA

    :return: None
    """

    # Ensure the dataset has a DEM variable
    assert 'dem' in ds.data_vars, "The DEM is missing."

    # Create a rasterio dataset in memory with the DEM data
    with rio.io.MemoryFile() as memfile:
        with memfile.open(
                driver='GTiff',
                height=ds.dem.data.shape[0],
                width=ds.dem.data.shape[1],
                count=1,
                dtype=ds.dem.data.dtype,
                crs=ds.rio.crs,
                transform=ds.rio.transform(),
        ) as dataset:
            dem_data = ds.dem.data.copy()

            # We expect DEMs without data gaps (NA values for the DEM features are not allowed in the data loading)
            assert np.sum(np.isnan(dem_data)) == 0, \
                "The DEM contains NA values. Please fill them before computing the DEM features."

            # Smooth the DEM with a 3x3 gaussian kernel (otherwise the DEM features are too noisy)
            dem_data = gaussian_filter(dem_data, sigma=1, mode='nearest', radius=1)

            # Prepare a xDEM object
            dataset.write(dem_data, 1)
            dem = xdem.DEM.from_array(dataset.read(1), transform=ds.rio.transform(), crs=ds.rio.crs)

            # Compute the DEM features
            attributes = xdem.terrain.get_terrain_attribute(dem.data, resolution=dem.res, attribute=xdem_features)

            # Add the features to the glacier dataset (remove the NANs from the margins)
            for k, data in zip(xdem_features, attributes):
                data_padded = np.pad(data[1:-1, 1:-1].astype(np.float32), pad_width=1, mode='edge')
                ds[k] = (('y', 'x'), data_padded)
                ds[k].attrs['_FillValue'] = np.float32(no_data)

    return ds
