import re
from pathlib import Path

import geopandas
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


def add_glacier_masks(
        nc_data,
        gl_df,
        entry_id_int,
        buffer=0,
        check_data_coverage=True,
        buffers_masks=(-20, -10, 0, 10, 20, 50)
):
    """
    Adds glacier masks to the given dataset after cropping it using the given buffer.

    The masks are:
        * mask_all_g_id: a mask with the IDs of all the glaciers in the current glacier's bounding box
        * mask_crt_g: a binary mask with the current glacier
        * mask_crt_g_bX: a binary mask with the current glacier and a buffer of X meters around it

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
    nc_data_crop['mask_all_g_id'] = (('y', 'x'), mask_rgi_id)
    nc_data_crop['mask_all_g_id'].attrs['_FillValue'] = -1
    nc_data_crop['mask_all_g_id'].rio.write_crs(nc_data.rio.crs, inplace=True)

    # 2. binary mask only for the current glacier, also with various buffers (in meters)
    for buffer_mask in buffers_masks:
        _crt_g_shp = crt_g_shp.buffer(buffer_mask).iloc[0]
        if not _crt_g_shp.is_empty:
            mask_crt_g = build_binary_mask(nc_data_crop, geoms=[_crt_g_shp])
        else:
            mask_crt_g = np.zeros_like(nc_data_crop.mask_all_g_id)

        label = '' if buffer_mask == 0 else f'_b{buffer_mask}'
        k = 'mask_crt_g' + label
        nc_data_crop[k] = (('y', 'x'), mask_crt_g.astype(np.int8))
        nc_data_crop[k].attrs['_FillValue'] = -1
        nc_data_crop[k].rio.write_crs(nc_data.rio.crs, inplace=True)

    return nc_data_crop


def prep_glacier_dataset(
        fp_img: str | Path,
        entry_id: str,
        gl_df: geopandas.GeoDataFrame,
        bands_name_map: dict | None = None,
        bands_qc_mask=None,
        extra_gdf_dict: dict | None = None,
        buffer_px: int = 0,
        check_data_coverage=True,
        fp_out: str | Path | None = None,
        return_nc: bool = False
):
    """
    Prepare a glacier dataset by adding the glacier masks and possibly extra masks (see add_glacier_masks).

    :param fp_img: the path to the raw image
    :param entry_id: the ID of the current glacier
    :param gl_df: the geopandas dataframe with all the glacier outlines
    :param bands_name_map: A dict containing the bands we keep from the raw data (as keys) and their new names (values);
        if None, all the bands are kept
    :param bands_qc_mask: the names of the bands to be used for building the mask for bad pixels
    :param extra_gdf_dict: a dictionary with the extra masks to be added
    :param buffer_px: the buffer around the current glacier (in pixels) to be used when cropping the image data
    :param check_data_coverage: whether to check if the data covers the current glacier + buffer
    :param fp_out: the path to the output glacier dataset (if None, the raster is returned without saving it)
    :param return_nc: whether to return the xarray dataset
    :return: None or the xarray dataset
    """
    row_crt_g = gl_df[gl_df.entry_id == entry_id]
    assert len(row_crt_g) == 1

    # read the raw image
    nc = xr.open_dataset(fp_img, mask_and_scale=False)

    # check if the name of the bands is given, otherwise name them
    if 'long_name' not in nc.band_data.attrs:
        nc.band_data.attrs['long_name'] = [f'B{i + 1}' for i in range(len(nc.band_data))]

    # set the band names for indexing
    nc = nc.assign_coords(band=list(nc.band_data.long_name))

    # build the mask for the bad (e.g. cloudy, shadowed or missing) pixels if needed
    if bands_qc_mask is not None:
        # ensure the bands to keep are in the image
        bands_missing = [b for b in bands_qc_mask if b.replace('~', '') not in nc.band_data.long_name]
        assert len(bands_missing) == 0, f"{bands_missing} not found in the image bands = {nc.band_data.long_name}"

        # build the mask
        mask_nok = np.zeros_like(nc.band_data.isel(band=0).values, dtype=bool)
        for b in bands_qc_mask:
            crt_mask = (nc.band_data.sel(band=b.replace('~', '')).values == 1)
            # invert the mask if needed
            if b[0] == '~':
                crt_mask = ~crt_mask
            mask_nok |= crt_mask

        # add the mask to the dataset
        nc['mask_nok'] = (('y', 'x'), mask_nok.astype(np.int8))
        nc['mask_nok'].attrs['_FillValue'] = -1
        nc['mask_nok'].attrs['bands_qc_mask'] = tuple(bands_qc_mask)
        nc['mask_nok'].rio.write_crs(nc.rio.crs, inplace=True)

    # keep only the bands we will need later if given
    if bands_name_map is not None:
        # ensure the bands to keep are in the image
        bands_missing = [b for b in bands_name_map if b not in nc.band_data.long_name]
        assert len(bands_missing) == 0, f"{bands_missing} not found in the image bands = {nc.band_data.long_name}"

        # subset the bands
        bands_to_keep = list(bands_name_map.keys())
        nc = nc.sel(band=list(bands_to_keep))

        # rename the bands
        new_band_names = [bands_name_map[b] for b in bands_to_keep]
        nc = nc.assign_coords(band=new_band_names)
        nc.band_data.attrs['long_name'] = new_band_names

    # add the glacier masks
    entry_id_int = row_crt_g.iloc[0].entry_id_i
    dx = nc.rio.resolution()[0]
    buffer = buffer_px * dx
    nc = add_glacier_masks(
        nc_data=nc,
        gl_df=gl_df,
        entry_id_int=entry_id_int,
        buffer=buffer,
        check_data_coverage=check_data_coverage
    )

    # add the extra masks if given
    if extra_gdf_dict is not None:
        for k, gdf in extra_gdf_dict.items():
            mask_name = f"mask_{k}"
            mask = build_binary_mask(nc, geoms=gdf.geometry.values)
            nc[mask_name] = (('y', 'x'), mask.astype(np.int8))
            nc[mask_name].attrs['_FillValue'] = -1
            nc[mask_name].rio.write_crs(nc.rio.crs, inplace=True)

    # not sure why but needed for QGIS
    nc['band_data'].rio.write_crs(nc.rio.crs, inplace=True)

    # export if needed
    if fp_out is not None:
        fp_out.parent.mkdir(exist_ok=True, parents=True)
        nc.attrs['fn'] = fp_img.name
        nc.attrs['glacier_area'] = row_crt_g.area_km2.iloc[0]
        nc.to_netcdf(fp_out)
        nc.close()

    if return_nc:
        return nc
    return None


def add_external_rasters(fp_gl, extra_rasters_bb_dict, no_data):
    """
    Adds extra rasters to the glacier dataset.

    :param fp_gl: str, path to the glacier dataset
    :param extra_rasters_bb_dict: dict, the paths to the extra rasters as keys and their bounding boxes that will be
    used to determine which of them intersect the current glacier
    :param no_data: the value to be used as NODATA

    :return: None
    """
    with xr.open_dataset(fp_gl, decode_coords='all', mask_and_scale=False) as nc_gl:
        nc_gl.load()  # needed to be able to close the file and save the changes to the same file
        nc_gl_bbox = shapely.geometry.box(*nc_gl.rio.bounds())

        # add a buffer to make sure we include all the rasters possibly intersecting the current glacier
        # (otherwise we may miss some due to re-projection errors)
        nc_gl_bbox = nc_gl_bbox.buffer(100)

        for k, crt_extra_rasters_bb_dict in extra_rasters_bb_dict.items():
            # check which raster files intersect the current glacier
            crt_nc_list = []
            for fp, raster_bbox in crt_extra_rasters_bb_dict.items():
                crt_nc = xr.open_dataarray(fp, mask_and_scale=False)
                transform = pyproj.Transformer.from_crs(crt_nc.rio.crs, nc_gl.rio.crs, always_xy=True).transform
                if nc_gl_bbox.intersects(shapely.ops.transform(transform, raster_bbox)):
                    crt_nc_list.append(crt_nc)

            # ensure at least one intersection was found
            assert len(crt_nc_list) > 0, f"No intersection found for {k} and fp_gl = {fp_gl}"

            # merge the datasets if needed
            nc_raster = rxr.merge.merge_arrays(crt_nc_list) if len(crt_nc_list) > 1 else crt_nc_list[0]

            # reproject
            nc_raster = nc_raster.isel(band=0).rio.reproject_match(
                nc_gl, resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
            nc_raster.rio.write_crs(nc_gl.rio.crs, inplace=True)  # not sure why but needed for QGIS
            nc_raster.attrs['_FillValue'] = np.float32(no_data)

            # add the current raster to the glacier dataset
            nc_gl[k] = nc_raster

    # export
    nc_gl.to_netcdf(fp_gl)
    nc_gl.close()


def add_dem_features(fp_gl, no_data):
    """
    Add DEM features to the glacier dataset using the XDEM library.
    The features are: slope, aspect, planform curvature, profile curvature, terrain ruggedness index.

    :param fp_gl: the path to the glacier dataset (the result will be saved in the same file)
    :param no_data: the value to be used as NODATA

    :return: None
    """
    # read the glacier dataset
    with xr.open_dataset(fp_gl, decode_coords='all', mask_and_scale=False) as nc_gl:
        nc_gl.load()  # needed to be able to close the file and save the changes to the same file
        assert 'dem' in nc_gl.data_vars, "The DEM is missing."

        # create a rasterio dataset in memory with the DEM data
        with rio.io.MemoryFile() as memfile:
            with memfile.open(
                    driver='GTiff',
                    height=nc_gl.dem.data.shape[0],
                    width=nc_gl.dem.data.shape[1],
                    count=1,
                    dtype=nc_gl.dem.data.dtype,
                    crs=nc_gl.rio.crs,
                    transform=nc_gl.rio.transform(),
            ) as dataset:
                dem_data = nc_gl.dem.data.copy()

                # we expect DEMs without data gaps (NA values for the DEM features are not allowed in the data loading)
                assert np.sum(np.isnan(dem_data)) == 0

                # smooth the DEM with a 3x3 gaussian kernel
                dem_data = gaussian_filter(dem_data, sigma=1, mode='nearest', radius=1)

                # prepare a XDEM object
                dataset.write(dem_data, 1)
                dem = xdem.DEM.from_array(dataset.read(1), transform=nc_gl.rio.transform(), crs=nc_gl.rio.crs)

                # compute the DEM features
                attrs_names = [
                    'slope',
                    'aspect',
                    'planform_curvature',
                    'profile_curvature',
                    'terrain_ruggedness_index'
                ]
                attributes = xdem.terrain.get_terrain_attribute(
                    dem.data, resolution=dem.res, attribute=attrs_names
                )

                # add the features to the glacier dataset (remove the NANs from the margins)
                for k, data in zip(attrs_names, attributes):
                    data_padded = np.pad(data[1:-1, 1:-1].astype(np.float32), pad_width=1, mode='edge')
                    nc_gl[k] = (('y', 'x'), data_padded)
                    nc_gl[k].attrs['_FillValue'] = np.float32(no_data)
                    nc_gl[k].rio.write_crs(nc_gl.rio.crs, inplace=True)

    # export to the same file
    nc_gl.to_netcdf(fp_gl)
    nc_gl.close()
