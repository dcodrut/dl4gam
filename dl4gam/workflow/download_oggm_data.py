import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import oggm.cfg
import oggm.core.gis
import oggm.shop.hugonnet_maps
import oggm.shop.millan22
import oggm.tasks
import oggm.workflow
import xarray as xr

log = logging.getLogger(__name__)


def main(
        geoms_fp: str | Path,
        working_dir: str | Path,
        gsd: int,
        border_m: int,
        dem_source: str = None,
        num_procs: int = 1,
):
    """
    Main function to download various datasets from OGGM which can later be used as additional features for training.
    :param geoms_fp: file path to the processed glacier outlines
    :param working_dir: where to store the OGGM glacier directories
    :param gsd: ground sampling distance (in meters)
    :param border_m: border size in meters to add around each glacier (should be large enough to sample patches from)
    :param dem_source: which DEM source to use for OGGM (set it to None for letting OGGM choose the best one)
    :param num_procs: number of processes to use for parallel processing
    :return: None
    """

    # Read the inventory outlines
    log.info(f"Reading the glacier inventory from {geoms_fp}")
    gdf = gpd.read_file(geoms_fp, layer='glacier_sel')

    log.info('Setting up OGGM')
    oggm.cfg.initialize(logging_level='WARNING')

    oggm.cfg.PATHS['working_dir'] = str(working_dir)
    oggm.cfg.PARAMS['use_rgi_area'] = True  # keep our coordinates
    oggm.cfg.PARAMS['use_intersects'] = False
    oggm.cfg.PARAMS['grid_dx_method'] = 'fixed'
    oggm.cfg.PARAMS['fixed_dx'] = gsd
    oggm.cfg.PARAMS['rgi_version'] = '70G'
    oggm.cfg.PARAMS['border'] = int(np.ceil(border_m / gsd))  # in grid cells
    oggm.cfg.PARAMS['map_proj'] = 'utm'
    oggm.cfg.PARAMS['use_multiprocessing'] = num_procs > 1
    oggm.cfg.PARAMS['mp_processes'] = num_procs
    oggm.cfg.PARAMS['continue_on_error'] = True  # continue processing even if some glaciers fail

    # Check if the CRS is WGS84
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError("Expected the glacier outlines to be in WGS84 (EPSG:4326) CRS.")

    # Create a dummy RGI7-style ID (~ RGI2000-v7.0-G-11-00040)
    gdf['rgi_id'] = gdf.entry_id_i.apply(lambda x: f'INV-{x:05d}')

    # Set dummy values for the RGI regions  # TODO: could use here the RGI boundaries and set them correctly
    gdf['o1region'] = '01'
    gdf['o2region'] = '01'

    log.info(f"Initializing OGGM glacier directories in {working_dir} with gsd={gsd}m and border={border_m}m")
    gdirs = oggm.workflow.init_glacier_directories(gdf)

    # Download the DEMs
    log.info(f"Download DEMs from {dem_source} for {len(gdirs)} glacier directories")
    oggm.workflow.execute_entity_task(
        task=oggm.tasks.define_glacier_region,
        gdirs=gdirs,
        source=dem_source
    )

    # Fill the voids in the DEMs and smooth it (we will need it for computing the DEM features later using xDEM)
    log.info("Filling the voids in the DEMs and smoothing them")
    oggm.workflow.execute_entity_task(
        task=oggm.core.gis.process_dem,
        gdirs=gdirs,
    )

    # Add the Hugonnet et al. 2021 dhdt maps
    log.info("Adding the Hugonnet et al. 2021 dhdt maps")
    oggm.workflow.execute_entity_task(
        task=oggm.shop.hugonnet_maps.hugonnet_to_gdir,
        gdirs=gdirs,
    )

    # Add the surface velocities and ice thicknesses from Milan et al. 2022
    log.info("Adding the surface velocities and ice thicknesses from Milan et al. 2022")
    oggm.workflow.execute_entity_task(
        task=oggm.shop.millan22.velocity_to_gdir,
        gdirs=gdirs,
    )

    # Re-write the CRS in the netcdf gridded file (not working in QGIS for some reason)
    for gdir in gdirs:
        fp = gdir.get_filepath('gridded_data')
        ds = xr.load_dataset(fp, decode_coords='all')
        ds.close()
        ds.rio.write_crs(ds.attrs['pyproj_srs'], inplace=True)
        ds.to_netcdf(fp)
