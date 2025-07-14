import logging
from typing import Optional

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import scipy
import shapely.geometry
import shapely.ops

from .parallel_utils import run_in_parallel

log = logging.getLogger(__name__)


def get_average_coords(geom: shapely.geometry.base.BaseGeometry) -> tuple:
    """
    Extract average coordinates for both Polygon and MultiPolygon geometries

    :param geom: Shapely geometry (Polygon or MultiPolygon)
    :return: Average coordinates as a tuple (longitude, latitude)
    """
    coords = []

    # Handle both geometry types
    if geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        # Collect coordinates from all polygons in the multipolygon
        for poly in geom.geoms:
            coords.extend(list(poly.exterior.coords))

    # Calculate the average coordinates
    coords_array = np.array(coords)
    return np.mean(coords_array, axis=0)


def get_interior_centroid(geom: shapely.geometry.base.BaseGeometry):
    """
    Return the centroid if it lies inside geom (Polygon or MultiPolygon),
    otherwise return the point on geom closest to the centroid.

    :param geom: Polygon or MultiPolygon
    :return: Point inside the geometry
    """

    if not isinstance(geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
        raise ValueError(f"Unsupported geometry type: {type(geom)}. Only Polygon and MultiPolygon are supported.")

    c = geom.centroid
    if geom.contains(c):
        return c

    return shapely.ops.nearest_points(c, geom)[1]


def add_country(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add the country column to the GeoDataFrame using representative points and the Natural Earth dataset.

    :param gdf: input GeoDataFrame
    :return: GeoDataFrame with the country column and codes
    """

    # Get the country boundaries from Natural Earth
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    gdf_world = gpd.read_file(url)

    # Assign the country name/code to each polygon based on a representative point
    # (we don't treat cases where a glacier spans multiple countries)
    rep_points = gdf.to_crs(gdf_world.crs).geometry.representative_point()
    joined = gpd.sjoin(
        gpd.GeoDataFrame(geometry=rep_points),
        gdf_world[['ADMIN', 'ISO_A2_EH', 'ISO_A3_EH', 'geometry']],
        how='left',
        predicate='within'
    )
    gdf['country_name'] = joined['ADMIN'].values
    gdf['country_code'] = joined['ISO_A2_EH'].values
    gdf['country_code_iso3'] = joined['ISO_A3_EH'].values

    return gdf


def latlon_to_utm_epsg(lat: float, lon: float) -> str:
    """
    Returns the UTM EPSG code for a given lon/lat.
    """

    # Compute zone number 1–60
    zone = int((lon + 180) // 6) + 1

    # Compute numeric EPSG code
    code = 32600 + zone if lat >= 0 else 32700 + zone

    return f"epsg:{code}"


def calculate_equal_area_buffer(
        gdf: gpd.GeoDataFrame,
        min_area_thr: Optional[float] = None,
        start_distance: float = 0.0,
        step: float = 10.0
) -> float:
    """
    Given a GeoDataFrame, we calculate the additional buffer distance (starting from the start_distance) that is
    required to obtain an area equal to the original area.
    We use only the polygons with an area larger than min_area_thr (in km²) to calculate the buffer, but we don't allow
    intersecting the smaller polygons (which is why we use the union of all geometries as starting point).

    :param gdf: GeoDataFrame with all polygons
    :param min_area_thr: minimum area threshold to consider a polygon (in km²) when calculating the buffer
    :param start_distance: initial buffer distance (in meters)
    :param step: step size for the buffer distance (in meters)

    :return: the buffer distance required to obtain an area equal to the original glacier area
    """

    # First let's simplify the polygons to make it faster;
    # We will also use a low resolution (1) which is enough for the purpose of this function
    _gdf = gdf.copy()
    _gdf['geometry'] = _gdf.geometry.simplify(step, preserve_topology=True)

    # Let's apply the initial buffer to all the geometries and union them (this region will always be subtracted later)
    geom_to_avoid = _gdf.geometry.buffer(start_distance, resolution=1).buffer(0).union_all()

    # Group the geometries into two sets based on the area threshold
    idx_large = (_gdf.area / 1e6 >= (min_area_thr if min_area_thr is not None else 0))
    gdf_l = _gdf[idx_large]  # these are the main ones we will use to calculate the non-overlapping buffer
    gdf_s = _gdf[~idx_large]  # we use these only to simulate the tessellation

    # Calculate the total area of the large polys (i.e. our target area)
    target_area = gdf_l.area.sum() / 1e6

    # Now we grow our polys, subtract the initial buffered region and check when we reach the target area
    buffer_dx = 0.0  # on top of the start_distance
    crt_area = 0.0
    while crt_area < target_area:
        buffer_dx += step

        # Buffer, union and subtract the initial buffered region
        crt_geom = gdf_l.buffer(start_distance + buffer_dx, resolution=1).union_all().difference(geom_to_avoid)

        # In the second step, we will simulate the tessellation by buffering the small polygons with half the distance
        if not gdf_s.empty:
            geom_to_avoid_s = gdf_s.buffer(start_distance + buffer_dx / 2, resolution=1).union_all()
            crt_geom = crt_geom.difference(geom_to_avoid_s)

        crt_area = crt_geom.area / 1e6
        log.debug(
            f"current buffer = {start_distance + buffer_dx}: "
            f"non-overlapping buffered area: {crt_area:.2f} km² (target: {target_area:.2f} km²)"
        )

    return start_distance + buffer_dx


def get_connected_components(gdf: gpd.GeoDataFrame, buffer_distance: float | int) -> np.ndarray:
    """
    Builds a symmetric adjacency matrix of overlapping glacier polygons compared to their buffers and extracts connected
    components.

    :param gdf: GeoDataFrame with glacier polygons
    :param buffer_distance: the buffer distance to apply to each polygon (in meters) before checking for overlaps
    :return: labels: array of labels for each polygon
    """

    # First, compute the pairs of polygons that overlap with their buffers
    _gdf_left = gdf.reset_index(drop=True).reset_index().rename(columns={'index': 'idx'})[['idx', 'geometry']]
    _gdf_right = gdf.reset_index(drop=True).reset_index().rename(columns={'index': 'idx'})[['idx', 'geometry']]
    max_buffer = buffer_distance * 2 + 1  # double the buffer distance to account for overlaps
    _gdf_right['geometry'] = _gdf_right.geometry.buffer(max_buffer)
    pairs = gpd.sjoin(_gdf_left, _gdf_right, how='inner', predicate='intersects')
    pairs = pairs[pairs.idx_left < pairs.idx_right]  # remove self-matches and duplicate pairs

    # Cluster the polygons based on the pairs
    left = pairs['idx_left'].to_numpy(dtype=int)
    right = pairs['idx_right'].to_numpy(dtype=int)
    data = np.ones(len(left), dtype=bool)
    adj = scipy.sparse.coo_matrix((data, (left, right)), shape=(len(gdf), len(gdf)))
    adj = adj + adj.T  # make it undirected
    _, labels = scipy.sparse.csgraph.connected_components(csgraph=adj, directed=False, return_labels=True)

    return labels


def buffer_non_overlapping(
        gdf: gpd.GeoDataFrame,
        buffer_distance: float | int,
        grid_size: float | int) -> gpd.GeoSeries:
    """
    Extend the outlines of the polygons in the GeoDataFrame by a specified buffer distance without overlapping.
    Based on https://github.com/geopandas/geopandas/issues/2015.

    :param gdf: GeoDataFrame with the glacier polygons
    :param buffer_distance: distance to extend the outlines
    :param grid_size: size of the grid to use for the voronoi tessellation (in meters)

    :return: GeoSeries with the non-overlapping buffers
    """

    # Ensure the data is in a projected CRS with metric units
    if not gdf.crs or gdf.crs.is_geographic:
        raise ValueError("GeoDataFrame must have a projected CRS with metric units")

    # If we have a single geometry, we can return a simple buffer
    if len(gdf) == 1:
        return gdf.buffer(buffer_distance, resolution=2).buffer(0)

    # Create a limit for the final combined buffers
    limit = gdf.buffer(buffer_distance, resolution=2).buffer(0).union_all()

    buffered_geoms = libpysal.cg.voronoi_frames(
        gdf,
        clip=limit,
        grid_size=grid_size,
        return_input=False,
        as_gdf=False,
    ).geometry.buffer(0)

    return buffered_geoms


def multi_buffer_non_overlapping_parallel(
        gdf: gpd.GeoDataFrame,
        buffers: list[float | int],
        num_procs: int = None
) -> list[gpd.GeoSeries]:
    """
    Calls buffer_non_overlapping (in parallel) after clustering the polygons into connected components.
    :param gdf: GeoDataFrame with the glacier polygons
    :param buffers: list of distances to extend the outlines for each buffer (in meters)
    :param num_procs: number of parallel processes to use (if None, we will use the default from run_in_parallel)
    :return:
    """

    # To get non-overlapping buffers, we will use momepy.morphological_tessellation
    # However, it the polygons are spatially dispersed, we run into memory issues (or takes too long).
    # So we will first cluster the polygons using connected components and then apply the tessellation in each cluster.
    # We run the clustering only once in the worst case scenario, i.e. when the buffer distance is the largest
    clustering_dist = np.max(buffers) * 2 + 1  # double the buffer distance to account for overlaps
    log.info(f"Clustering polygons into connected components with buffer distance {clustering_dist} m")
    clusters = get_connected_components(gdf=gdf, buffer_distance=clustering_dist)

    # Run buffer_non_overlapping in parallel for each cluster
    all_buffered_geoms = []
    for buffer_distance in buffers:
        log.info(f"Computing non-overlapping buffers for {buffer_distance} m")
        gdf_per_cluster = [gdf[clusters == i] for i in range(np.max(clusters) + 1)]
        assert sum(len(gdf_per_cluster[i]) for i in range(len(gdf_per_cluster))) == len(gdf), \
            "The clustering did not cover all polygons."

        # Start with the largest clusters for a better balance of workload
        gdf_per_cluster = sorted(gdf_per_cluster, key=lambda x: len(x), reverse=True)

        res = run_in_parallel(
            fun=buffer_non_overlapping,
            pbar_desc=f"Buffering polygons with {buffer_distance} m",
            gdf=gdf_per_cluster,
            buffer_distance=buffer_distance,
            num_procs=num_procs,
            gc_collect_step=100  # Ran into some memory issues
        )
        # Now concatenate the results and put them back in order
        buffered_geom = gpd.GeoSeries(pd.concat(res, ignore_index=False), crs=gdf.crs)
        buffered_geom = buffered_geom[gdf.index]
        all_buffered_geoms.append(buffered_geom)

    return all_buffered_geoms
