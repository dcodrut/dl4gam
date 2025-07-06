import logging

import geopandas as gpd
import momepy
import numpy as np
import shapely.geometry
import shapely.ops

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


def calculate_equal_area_buffer(gdf: gpd.GeoDataFrame, start_distance: float = 0.0, step: float = 10.0) -> float:
    """
    Given a GeoDataFrame, we calculate the additional buffer distance (starting from the start_distance) that is
    required to obtain an area equal to the original area.

    :param gdf: GeoDataFrame with the polygons to be buffered
    :param start_distance: initial buffer distance (in meters)
    :param step: step size for the buffer distance (in meters)

    :return: the buffer distance required to obtain an area equal to the original glacier area
    """

    # first let's simplify the polygons to make it faster
    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.simplify(step, preserve_topology=True)

    log.debug("Applying union_all to polygons")
    init_geom = gdf.geometry.union_all()
    initial_area = init_geom.area / 1e6

    log.debug(f"Applying starting buffer of {start_distance:.2f} m")
    init_geom = init_geom.buffer(start_distance, join_style=2, resolution=1)
    buffer_distance = start_distance
    crt_buffer_area = init_geom.area / 1e6

    while crt_buffer_area < 2 * initial_area:
        buffer_distance += step

        # buffer the union (to avoid double counting)
        log.debug(f"Applying buffer of {buffer_distance:.2f} m")
        crt_geom = init_geom.buffer(buffer_distance, join_style=2, resolution=1)
        crt_buffer_area = crt_geom.area / 1e6
        log.debug(f"Buffer area: {crt_buffer_area:.2f} km²")

    return buffer_distance


def buffer_non_overlapping(gdf: gpd.GeoDataFrame, buffer_distance: float) -> gpd.GeoSeries:
    """
    Extend the outlines of the polygons in the GeoDataFrame by a specified buffer distance without overlapping.
    Based on https://github.com/geopandas/geopandas/issues/2015.

    :param gdf: GeoDataFrame with the glacier polygons
    :param buffer_distance: distance to extend the outlines

    :return: GeoSeries with the non-overlapping buffers
    """

    # Ensure the data is in a projected CRS with metric units
    if not gdf.crs or gdf.crs.is_geographic:
        raise ValueError("GeoDataFrame must have a projected CRS with metric units")

    # Create a limit for the final combined buffers
    limit = gdf.union_all().buffer(buffer_distance, join_style=2, resolution=1)

    # Build the non-overlapping Voronoi polygons
    buffered_geoms = momepy.morphological_tessellation(gdf, clip=limit, simplify=False).geometry

    return buffered_geoms
