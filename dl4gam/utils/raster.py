import geopandas as gpd
import numpy as np
import rasterio.features
import shapely.geometry
import shapely.ops
import xarray as xr


def polygonize(da: xr.DataArray, min_segment_area_km2: float = 0.0, connectivity=8):
    """
    Polygonize a raster DataArray, returning a GeoDataSeries the (multi)polygons.
    :param da: DataArray with a binary mask (1 for the area of interest, 0 for the background), in a projected CRS.
    :param min_segment_area_km2: minimum area of the polygons to keep; polygons smaller than this will be discarded
    :param connectivity: connectivity for the polygonization (see rasterio.features.shapes documentation)
    :return: GeoDataSeries with (multi)polygons
    """

    if da.dtype not in ['uint8', 'bool']:
        raise ValueError(f"DataArray must be of type 'uint8' or 'bool', got {da.dtype}")

    da = da.astype('uint8')  # rasterio.features.shapes doesn't allow bools

    # Get the shapes and covert them to polygons
    shapes = list(rasterio.features.shapes(da, transform=da.rio.transform(), connectivity=connectivity))
    geometries = [shapely.geometry.shape(shape) for shape, value in shapes if value == 1]

    # Filter out geometries smaller than the minimum area
    geometries = [geom for geom in geometries if geom.area / 1e6 >= min_segment_area_km2]

    # Create multipolygon or empty poly if no shapes found
    multipoly = shapely.ops.unary_union(geometries) if len(geometries) > 0 else shapely.geometry.Polygon()

    # Create GeoSeries with original CRS and reproject to WGS84
    gs = gpd.GeoSeries([multipoly], crs=da.rio.crs)
    gs_wgs84 = gs.to_crs(epsg=4326)

    # Return the (muli)poly only
    return gs_wgs84.iloc[0]


def nn_interp(data, mask_to_fill, mask_ok, num_nn=100):
    y_px_to_fill, x_px_to_fill = np.where(mask_to_fill)
    y_px_ok, x_px_ok = np.where(mask_ok)

    data_interp = np.zeros_like(data) + np.nan
    data_interp[mask_ok] = data[mask_ok]
    n_ok = len(x_px_ok)
    for x, y in zip(x_px_to_fill, y_px_to_fill):
        # get the closest num_nn pixels
        dists = (x - x_px_ok) ** 2 + (y - y_px_ok) ** 2

        # keep only the closest pixels
        max_dist = np.quantile(dists, q=num_nn / n_ok)
        idx = (dists <= max_dist)
        x_px_ok_sel, y_px_ok_sel = x_px_ok[idx], y_px_ok[idx]

        # compute the mean over the selected pixels
        fill_value = np.mean(data[y_px_ok_sel, x_px_ok_sel])
        data_interp[y, x] = fill_value
    return data_interp


def hypso_interp(data, mask_to_fill, mask_ok, dem, num_px=100):
    # get the unique elevation values that have to be filled in
    h_to_fill_sorted = np.sort(np.unique(dem[mask_to_fill]))

    # prepare a sorted array of filled elevations which will be used to get the interpolation
    h_ok_sorted = np.sort(dem[mask_ok])

    data_interp = np.zeros_like(data) + np.nan
    data_interp[mask_ok] = data[mask_ok]
    for h in h_to_fill_sorted:
        # get the closest ~num_px based on their elevation
        i_h = np.searchsorted(h_ok_sorted, h)
        i_h_min = max(0, i_h - num_px // 2)
        i_h_max = min(len(h_ok_sorted) - 1, i_h + num_px // 2)
        h_min = h_ok_sorted[i_h_min]
        h_max = h_ok_sorted[i_h_max]

        crt_mask_h_ok = mask_ok & (h_min <= dem) & (dem <= h_max)
        crt_mask_h_to_fill = mask_to_fill & (dem == h)
        fill_value = np.mean(data[crt_mask_h_ok])
        data_interp[crt_mask_h_to_fill] = fill_value

    return data_interp
