import functools
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.ops
import xarray as xr

# local imports
from config import C
from utils.data_prep import prep_glacier_dataset, add_external_rasters, add_dem_features
from utils.data_stats import compute_qc_stats
from utils.general import run_in_parallel


def add_stats(gl_df, fp_stats, num_procs, col_clouds, col_ndsi, col_albedo):
    """
        Add the image statistics (i.e. cloud and/or shadow, NDSI and albedo) to the selected glacier dataframe.
        The function calls the compute_qc_stats function in parallel for each glacier which computes various statistics,
        both for the glacier and the surrounding area.

    :param gl_df: the glaciers dataframe
    :param fp_stats: the filepath to the csv file where to save the QC statistics (or read them from if it exists)
    :param num_procs: the number of parallel processes to use for computing the stats
    :param col_clouds: the column name for the cloud coverage (including shadows and missing pixels)
    :param col_ndsi: the column name for the NDSI
    :param col_albedo: the column name for the albedo
    :return: the updated dataframe with the QC statistics
    """

    if fp_stats.exists():
        print(f"Reading the cloud & shadow stats from {fp_stats}")
        cloud_stats_df = pd.read_csv(fp_stats)
    else:
        gl_sdf_list = list(
            gl_df.apply(lambda r: gpd.GeoDataFrame(pd.DataFrame(r).T, crs=gl_df.crs), axis=1)
        )
        all_cloud_stats = run_in_parallel(
            fun=functools.partial(compute_qc_stats, include_shadows=False),
            gl_sdf=gl_sdf_list,
            num_procs=num_procs,
            pbar=True
        )
        cloud_stats_df = pd.DataFrame.from_records(all_cloud_stats)
        cloud_stats_df = cloud_stats_df.sort_values(['entry_id', 'fp_img'])
        fp_stats.parent.mkdir(exist_ok=True, parents=True)
        cloud_stats_df.to_csv(fp_stats, index=False)
        print(f"cloud stats exported to {fp_stats}")
    cloud_stats_df.fp_img = cloud_stats_df.fp_img.apply(lambda s: Path(s))

    cols_qc = [col_clouds, col_ndsi, col_albedo]
    gl_df = gl_df.merge(cloud_stats_df[['fp_img'] + cols_qc], on='fp_img')

    return gl_df


def select_best_images(gl_df, fp_stats_all, fp_stats_selected, col_clouds, col_ndsi, col_albedo, max_n_imgs_per_g):
    """
    Select the best images based on the cloud coverage and the NDSI, plus the albedo as a tie-breaker.

    :param gl_df: the glacier dataframe containing already the image statistics
    :param fp_stats_all: the filepath to the csv file where to save the QC statistics for all images
    :param fp_stats_selected: the filepath to the csv file where to save the selected images
    :param col_clouds: the column name for the cloud and/or shadow coverage
    :param col_ndsi: the column name for the NDSI
    :param col_albedo: the column name for the albedo
    :param max_n_imgs_per_g: the maximum number of images to keep per glacier
    :return:
    """
    # give an index to each image based on the cloud coverage and the NDSI
    for c in [col_clouds, col_ndsi]:
        df_idx = gl_df.groupby('entry_id', sort=True, group_keys=False)[['entry_id', 'date', c]].apply(
            lambda ssdf: pd.DataFrame(
                {
                    'entry_id': ssdf.entry_id,
                    'date': ssdf.date,
                    f"{c}_s": np.searchsorted(ssdf[c].round(2).sort_values(), ssdf[c].round(2)),
                }
            )
        )

        gl_df = gl_df.merge(df_idx, on=['entry_id', 'date'], validate='one_to_one')

    # compute the combined score based on the cloud coverage and the NDSI
    gl_df['qc_score'] = (gl_df[f"{col_clouds}_s"] + gl_df[f"{col_ndsi}_s"]) / 2
    gl_df = gl_df.sort_values(['entry_id', 'date'])

    # save the scores to a csv file
    pd.DataFrame(gl_df.drop(columns='geometry')).to_csv(fp_stats_all, index=False)
    print(f"QC stats exported to {fp_stats_all}")

    # keep the best (use albedo as tie-breaker) & export the selected dataframe
    gl_df = gl_df.sort_values(['qc_score', col_albedo])
    gl_sdf = gl_df.groupby('entry_id').head(max_n_imgs_per_g).reset_index()
    gl_sdf = gl_sdf.sort_values(['entry_id', 'date'])
    pd.DataFrame(gl_sdf.drop(columns='geometry')).to_csv(fp_stats_selected, index=False)
    print(f"Selected images exported to {fp_stats_selected}")

    return gl_sdf


def prepare_all_rasters(
        fp_gl_df_all,
        out_rasters_dir,
        buffer_px,
        check_data_coverage=True,
        raw_images_dir=None,
        raw_fp_df=None,
        no_data=C.NODATA,
        num_procs=1,
        bands_to_keep='all',
        extra_geometries_dict=None,
        extra_rasters_dict=None,
        min_area=None,
        df_dates=None,
        choose_best_auto=False,
        max_cloud_f=None,
        max_n_imgs_per_g=1,
        compute_dem_features=False
):
    """
        Prepare the rasters for all the glaciers in the given dataframe.

    :param fp_gl_df_all: the filepath to the shapefile containing the glacier outlines
    :param out_rasters_dir: the directory where to save the glacier-wide raster files
    :param buffer_px: the buffer in pixels to add around the glacier outlines
    :param check_data_coverage: if True, check if the raw image has a large enough spatial extent
    :param raw_images_dir: the directory containing the raw images (optional, if raw_fp_df is not given)
    :param raw_fp_df: the dataframe containing the list of images for each glacier (optional, if raw_images_dir is not given)
    :param no_data: the no data value to use for the output rasters
    :param num_procs: the number of parallel processes to use
    :param bands_to_keep: the bands to keep from the raw images
    :param extra_geometries_dict: a dictionary containing the extra shapefiles to add to the glacier datasets as binary masks
    :param extra_rasters_dict: a dictionary containing the extra rasters to add to the glacier datasets
    :param min_area: the minimum glacier area to consider (only the rasters for these glaciers will be built but the main binary masks will contain all glaciers)
    :param df_dates: a dataframe containing the allowed dates for each glacier (if None, all the images will be used)
    :param choose_best_auto: if True, keep the best images based on the cloud coverage and the NDSI
    :param max_cloud_f: the maximum cloud coverage allowed for each image
    :param max_n_imgs_per_g: the maximum number of images to keep per glacier
    :param compute_dem_features: if True, compute the features from the DEM (see the add_dem_features function)
    :return: None
    """

    print(f"Reading the glacier outlines in from {fp_gl_df_all}")
    gl_df_all = gpd.read_file(fp_gl_df_all)
    print(f"#glaciers = {len(gl_df_all)}")

    # check if the dataframes have the required ID columns
    assert 'entry_id' in gl_df_all.columns
    assert 'entry_id_i' in gl_df_all.columns

    if min_area is not None:
        # keep the glaciers of interest in a different dataframe
        gl_df_sel = gl_df_all[gl_df_all.area_km2 >= min_area]
        print(f"#glaciers = {len(gl_df_sel)} after area filtering")
    else:
        gl_df_sel = gl_df_all

    assert (raw_images_dir is not None) or (raw_fp_df is not None), \
        "Either raw_images_dir or raw_fp_df must be given (the latter has priority)."

    if raw_fp_df is None:
        # get all the raw images and match them to the corresponding glaciers if the raw_fp_df is not given
        raw_images_dir = Path(raw_images_dir)
        assert raw_images_dir.exists(), f"raw_images_dir = {raw_images_dir} not found."
        fp_img_list_all = sorted(list(raw_images_dir.glob('**/*.tif')))
        raw_fp_df = pd.DataFrame({
            'entry_id': [fp.parent.name for fp in fp_img_list_all],
            'fp_img': fp_img_list_all,
            'date': [pd.to_datetime(fp.stem[:8]).strftime('%Y-%m-%d') for fp in fp_img_list_all]
        })
    print(f"#total raw images = {len(raw_fp_df)}")

    if df_dates is not None:
        raw_fp_df = raw_fp_df.merge(df_dates, on=['entry_id', 'date'])
        print(f"#total raw images (after filtering by date) = {len(raw_fp_df)}")

    # add the filepaths to the selected glaciers and check the data coverage
    gid_list = set(gl_df_sel.entry_id)
    gl_df_sel = gl_df_sel.merge(raw_fp_df, on='entry_id', how='inner')  # we allow missing images
    print(f"avg. #images/glacier = {gl_df_sel.groupby('entry_id').count().fp_img.mean():.1f}")
    print(f"#glaciers with no data = {(n := len(gid_list - set(gl_df_sel.entry_id)))} ({n / len(gid_list) * 100:.1f}%)")

    # compute the cloud coverage statistics for each downloaded image if needed
    if choose_best_auto or max_cloud_f is not None:
        # specify the columns for the cloud coverage, the NDSI and the albedo stats (see the compute_qc_stats function)
        # the v1 suffix means we use the statistics based on the CLOUDLESS_MASK band
        # (i.e. Mask of filled & cloud/shadow-free pixels,
        # see https://geedim.readthedocs.io/en/latest/cli.html#geedim-download)
        col_clouds = 'cloud_p_gl_b50m_v1'
        col_ndsi = 'ndsi_avg_non_gl_b50m_v1'
        col_albedo = 'albedo_avg_gl_b50m_v1'

        fp_stats_all = Path(out_rasters_dir).parent / 'aux_data' / 'stats_all.csv'
        gl_df_sel = add_stats(
            gl_df=gl_df_sel,
            fp_stats=fp_stats_all,
            num_procs=num_procs,
            col_clouds=col_clouds,
            col_ndsi=col_ndsi,
            col_albedo=col_albedo
        )

        # impose the max cloudiness threshold if given (this includes missing pixels)
        if max_cloud_f is not None:
            gl_df_sel = gl_df_sel[gl_df_sel[col_clouds] <= max_cloud_f]
            print(f"after cloud filtering with max_cloud_f = {max_cloud_f:.2f}")
            print(f"\t#images = {len(gl_df_sel)}")
            print(f"\tavg. #images/glacier = {gl_df_sel.groupby('entry_id').count().fp_img.mean():.1f}")
            print(f"\t#glaciers with no data = {(n := len(gid_list - set(gl_df_sel.entry_id)))} "
                  f"({n / len(gid_list) * 100:.1f}%)")

        # keep the best n images (based on cloud coverage and NDSI, plus albedo as tie-breaker) if needed
        if choose_best_auto:
            fp_qc_stats_all = Path(out_rasters_dir).parent / 'aux_data' / 'qc_stats_all.csv'
            fp_qc_stats_selected = Path(out_rasters_dir).parent / 'aux_data' / 'qc_stats_selected.csv'

            gl_df_sel = select_best_images(
                gl_df=gl_df_sel,
                fp_stats_all=fp_qc_stats_all,
                fp_stats_selected=fp_qc_stats_selected,
                col_clouds=col_clouds,
                col_ndsi=col_ndsi,
                col_albedo=col_albedo,
                max_n_imgs_per_g=max_n_imgs_per_g
            )
            print(f"after keeping the best {max_n_imgs_per_g} images per glacier:")
            print(f"\t#images = {len(gl_df_sel)}")
            print(f"\tavg. #images/glacier = {gl_df_sel.groupby('entry_id').count().fp_img.mean():.1f}")
            print(f"\t#glaciers with no data = {(n := len(gid_list - set(gl_df_sel.entry_id)))} "
                  f"({n / len(gid_list) * 100:.1f}%)")
            print(f"len(gl_df_sel) = {len(gl_df_sel)}")

    # read the extra shapefiles
    if extra_geometries_dict is not None:
        extra_gdf_dict = {}
        for k, fp in extra_geometries_dict.items():
            print(f"Reading the outlines {k} from {fp}")
            extra_gdf_dict[k] = gpd.read_file(fp)
    else:
        extra_gdf_dict = None

    # prepare the output paths
    fp_img_list = [Path(x) for x in gl_df_sel.fp_img]
    fn_img_out_list = [x.with_suffix('.nc').name for x in fp_img_list]
    fp_out_list = [Path(out_rasters_dir) / x / y for x, y in zip(gl_df_sel.entry_id, fn_img_out_list)]

    # build and export the rasters for each image
    run_in_parallel(
        fun=prep_glacier_dataset,
        fp_img=fp_img_list,
        fp_out=fp_out_list,
        entry_id=list(gl_df_sel.entry_id),
        gl_df=gl_df_all,
        extra_gdf_dict=extra_gdf_dict,
        bands_to_keep=bands_to_keep,
        buffer_px=buffer_px,
        check_data_coverage=check_data_coverage,
        no_data=no_data,
        num_procs=num_procs,
        pbar=True
    )

    # save the bounding boxes of all the provided rasters (needed for the next step)
    extra_rasters_bb_dict = {}
    for k, crt_dir in extra_rasters_dict.items():
        rasters_crt_dir = list(Path(crt_dir).glob('**/*.tif'))
        extra_rasters_bb_dict[k] = {
            fp: shapely.geometry.box(*xr.open_dataset(fp).rio.bounds()) for fp in rasters_crt_dir
        }

    # add the extra rasters to the glacier datasets by loading those that intersect each glacier using the previous bb
    run_in_parallel(
        fun=add_external_rasters,
        fp_gl=fp_out_list,
        extra_rasters_bb_dict=extra_rasters_bb_dict,
        num_procs=num_procs,
        no_data=no_data,
        pbar=True
    )

    # derive features from the DEM if needed
    if compute_dem_features:
        run_in_parallel(
            fun=add_dem_features,
            fp_gl=fp_out_list,
            num_procs=num_procs,
            no_data=no_data,
            pbar=True
        )


if __name__ == "__main__":
    # the next settings apply to all datasets
    base_settings = dict(
        raw_images_dir=C.RAW_DATA_DIR,
        raw_fp_df=None,
        fp_gl_df_all=C.GLACIER_OUTLINES_FP,
        out_rasters_dir=C.DIR_GL_INVENTORY,
        min_area=C.MIN_GLACIER_AREA,
        bands_to_keep=C.BANDS,
        no_data=C.NODATA,
        num_procs=C.NUM_PROCS,
        extra_geometries_dict=C.EXTRA_GEOMETRIES,
        extra_rasters_dict=C.EXTRA_RASTERS,
        compute_dem_features=True
    )

    for k, v in {**C.EXTRA_GEOMETRIES, **C.EXTRA_RASTERS}.items():
        assert Path(v).exists(), f"{k} filepath: {v} does not exist"

    # some dataset specific settings
    specific_settings = {}
    if C.__name__ == 'S2_ALPS':
        specific_settings = dict(
            choose_best_auto=True,
            buffer_px=C.PATCH_RADIUS,
        )
    elif C.__name__ == 'S2_ALPS_PLUS':
        if C.CSV_DATES_ALLOWED is not None:
            dates_allowed = pd.read_csv(C.CSV_DATES_ALLOWED)
            max_cloud_f = None
            choose_best_auto = False
        else:
            dates_allowed = None
            max_cloud_f = 0.3
            choose_best_auto = True
        specific_settings = dict(
            buffer_px=C.PATCH_RADIUS,
            df_dates=dates_allowed,
            max_cloud_f=max_cloud_f,
            choose_best_auto=choose_best_auto
        )
    elif C.__name__ == 'S2_SGI':
        if C.CSV_DATES_ALLOWED is not None:
            print(f"Reading the allowed dates csv from {C.CSV_DATES_ALLOWED}")
            dates_allowed = pd.read_csv(C.CSV_DATES_ALLOWED)
            choose_best_auto = False
        else:
            dates_allowed = None
            choose_best_auto = True
        specific_settings = dict(
            choose_best_auto=choose_best_auto,
            max_cloud_f=0.3,
            buffer_px=C.PATCH_RADIUS,
            df_dates=dates_allowed
        )

    settings = base_settings.copy()
    settings.update(specific_settings)
    prepare_all_rasters(**settings)
