import numpy as np
import oggm
import oggm.core.gis
import pyproj
import rasterio
import rasterio as rio
import rioxarray as rxr
import rioxarray.merge
import shapely
import shapely.ops
import xarray as xr
import xdem


def add_glacier_masks(nc_data, gl_df, entry_id_int, buffer=0, buffers_masks=(-20, -10, 0, 10, 20, 50)):
    # project the glaciers' outlines on the same CRS as the given dataset
    gl_proj_df = gl_df.to_crs(nc_data.rio.crs)

    # get the outline of the current glacier
    crt_g_shp = gl_proj_df[gl_proj_df.entry_id_i == entry_id_int]

    # create a bounding box which contains any possible patch which will be sampled around the current glacier
    g_bbox = shapely.geometry.box(*crt_g_shp.iloc[0].geometry.bounds)
    g_buff = g_bbox.buffer(buffer)
    g_buff_bbox = shapely.geometry.box(*g_buff.bounds)

    # check if the bounding box with the buffer is completely contained in the data boundaries
    data_bbox = shapely.geometry.box(*nc_data.rio.bounds())
    assert data_bbox.contains(g_buff_bbox), f"The data does not include the current glacier and a {buffer}m buffer"

    # keep only the current glacier and its neighbours
    nc_data_crop = nc_data.rio.clip([g_buff_bbox])

    # add masks (current glacier mask & all glaciers mask, i.e. with glaciers IDs)
    # 1. all glaciers mask - with glaciers IDs
    mask_rgi_id = np.zeros_like(nc_data_crop.band_data.values[0], dtype=np.int32) - 1
    gl_crt_buff_df = gl_proj_df[gl_proj_df.intersects(g_buff_bbox)]
    for i in range(len(gl_crt_buff_df)):
        row = gl_crt_buff_df.iloc[i]
        tmp_raster = nc_data_crop.band_data.isel(band=0).fillna(0).rio.clip([row.geometry], drop=False).values
        mask_crt_g = ~np.isnan(tmp_raster)
        mask_rgi_id[mask_crt_g] = row.entry_id_i
    nc_data_crop['mask_all_g_id'] = (('y', 'x'), mask_rgi_id)
    nc_data_crop['mask_all_g_id'].attrs['_FillValue'] = -1
    nc_data_crop['mask_all_g_id'].rio.write_crs(nc_data.rio.crs, inplace=True)

    # 2. binary mask only for the current glacier, also with various buffers (in meters)
    for buffer_mask in buffers_masks:
        _crt_g_shp = crt_g_shp.buffer(buffer_mask)
        if not _crt_g_shp.iloc[0].is_empty:
            tmp_raster = nc_data_crop.band_data.isel(band=0).fillna(0).rio.clip(_crt_g_shp.geometry, drop=False).values
            mask_crt_glacier = (~np.isnan(tmp_raster)).astype(np.int8)
        else:
            mask_crt_glacier = np.zeros_like(nc_data_crop.mask_all_g_id).astype(np.int8)

        label = '' if buffer_mask == 0 else f'_b{buffer_mask}'
        k = 'mask_crt_g' + label
        nc_data_crop[k] = (('y', 'x'), mask_crt_glacier)
        nc_data_crop[k].attrs['_FillValue'] = -1
        nc_data_crop[k].rio.write_crs(nc_data.rio.crs, inplace=True)

    return nc_data_crop


def add_extra_mask(nc_data, mask_name, gdf):
    # project the outlines on the same CRS as the image data
    gdf_proj = gdf.to_crs(nc_data.rio.crs)

    # get all the polys that intersect the current raster (multiple glaciers can be covered)
    tmp_raster = nc_data.band_data.isel(band=0).rio.clip(gdf_proj.geometry, drop=False).values
    mask = (~np.isnan(tmp_raster)).astype(np.int8)
    nc_data[mask_name] = (('y', 'x'), mask)
    nc_data[mask_name].attrs['_FillValue'] = -1
    nc_data[mask_name].rio.write_crs(nc_data.rio.crs, inplace=True)

    return nc_data


def prep_glacier_dataset(
        fp_img, fp_out, entry_id, gl_df, extra_gdf_dict, buffer_px, no_data, bands_to_keep=None
):
    row_crt_g = gl_df[gl_df.entry_id == entry_id]
    assert len(row_crt_g) == 1

    # read the raw image
    nc = xr.open_dataset(fp_img)

    # check if the name of the bands is given, otherwise name them
    if 'long_name' not in nc.band_data.attrs:
        nc.band_data.attrs['long_name'] = [f'B{i + 1}' for i in range(len(nc.band_data))]

    # keep only the bands we need later if specified
    if bands_to_keep is not None:
        # ensure the bands to keep are in the image
        bands_missing = [b for b in bands_to_keep if b not in nc.band_data.long_name]
        assert len(bands_missing) == 0, f"Missing bands: {bands_missing}"

        all_bands = list(nc.band_data.long_name)
        nc = nc.isel(band=[all_bands.index(b) for b in bands_to_keep])
        nc.band_data.attrs['long_name'] = tuple(bands_to_keep)

    # add the glacier masks
    entry_id_int = row_crt_g.iloc[0].entry_id_i
    dx = nc.rio.resolution()[0]
    buffer = buffer_px * dx
    nc = add_glacier_masks(nc_data=nc, gl_df=gl_df, entry_id_int=entry_id_int, buffer=buffer)

    # add the extra masks if given
    if extra_gdf_dict is not None:
        for k, gdf in extra_gdf_dict.items():
            nc = add_extra_mask(nc_data=nc, mask_name=f"mask_{k}", gdf=gdf)

    # convert the image data to int16
    nc['band_data'] = nc.band_data.fillna(no_data).astype(np.int16)
    nc['band_data'].attrs['_FillValue'] = no_data
    nc['band_data'].rio.write_crs(nc.rio.crs, inplace=True)  # not sure why but needed for QGIS

    # export
    fp_out.parent.mkdir(exist_ok=True, parents=True)
    nc.attrs['fn'] = fp_img.name
    nc.attrs['glacier_area'] = row_crt_g.area_km2.iloc[0]
    nc.to_netcdf(fp_out)
    nc.close()


def add_external_rasters(fp_gl, extra_rasters_bb_dict, no_data):
    """
        Adds extra rasters to the glacier dataset.

        :param fp_gl: str, path to the glacier dataset
        :param extra_rasters_bb_dict: dict, the paths to the extra rasters as keys and their bounding boxes that will be
        used to determine which of them intersect the current glacier
        :no_data: the value to be used as NODATA
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
       :no_data: the value to be used as NODATA
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
                dem_data = oggm.core.gis.gaussian_blur(dem_data, size=1)

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
