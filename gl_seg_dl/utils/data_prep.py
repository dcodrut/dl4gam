import numpy as np
import shapely
import config as C


def add_glacier_masks(nc_data, gl_df, entry_id_int, buffer=0):
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
    nc_data_crop['mask_all_g_id'].attrs['_FillValue'] = C.S2.NODATA
    nc_data_crop['mask_all_g_id'].rio.write_crs(nc_data.rio.crs, inplace=True)

    # 2. binary mask only for the current glacier, also with various buffers (in meters)
    for buffer_mask in [0, 10, 20, 50]:
        _crt_g_shp = crt_g_shp.buffer(buffer_mask)
        tmp_raster = nc_data_crop.band_data.isel(band=0).fillna(0).rio.clip(_crt_g_shp.geometry, drop=False).values
        mask_crt_glacier = (~np.isnan(tmp_raster)).astype(np.int8)
        label = '' if buffer_mask == 0 else f'_b{buffer_mask}'
        k = 'mask_crt_g' + label
        nc_data_crop[k] = (('y', 'x'), mask_crt_glacier)
        nc_data_crop[k].attrs['_FillValue'] = -1
        nc_data_crop[k].rio.write_crs(nc_data.rio.crs, inplace=True)

    return nc_data_crop
