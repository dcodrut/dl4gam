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


def add_stats(
        gl_df: gpd.GeoDataFrame,
        bands_name_map: dict | None,
        bands_qc_mask: list,
        fp_stats: Path,
        num_procs: int,
        col_clouds: str,
        col_ndsi: str,
        col_albedo: str,
        buffer_px: int = 0
) -> gpd.GeoDataFrame:
    """
    Add the image statistics (i.e. cloud and/or shadow, NDSI and albedo) to the selected glacier dataframe.
    The function calls the compute_qc_stats function in parallel for each glacier which computes various statistics,
    both for the glacier and the surrounding area.

    :param gl_df: the glaciers dataframe
    :param bands_name_map: A dict containing the bands we keep from the raw data (as keys) and their new names (values).
        If None, all the bands will be kept.
    :param bands_qc_mask: the list of bands to use for the QC mask (e.g. the cloud coverage)
    :param fp_stats: the filepath to the csv file where to save the QC statistics (or read them from if it exists)
    :param num_procs: the number of parallel processes to use for computing the stats
    :param col_clouds: the column name for the cloud coverage (including shadows and missing pixels)
    :param col_ndsi: the column name for the NDSI
    :param col_albedo: the column name for the albedo
    :param buffer_px: the buffer in pixels to consider around the glacier outlines when cutting the raw images (it will
        impact the scene level statistics)

    :return: the updated dataframe with the QC statistics
    """

    if fp_stats.exists():
        print(f"Reading the cloud & shadow stats from {fp_stats}")
        cloud_stats_df = pd.read_csv(fp_stats)

        # make sure all the glacier IDs are present in the stats dataframe
        assert set(gl_df.entry_id).issubset(set(cloud_stats_df.entry_id)), \
            "Some glacier IDs are missing from the stats dataframe. Delete the file and rerun the script."
    else:
        gl_sdf_list = list(
            gl_df.apply(lambda r: gpd.GeoDataFrame(pd.DataFrame(r).T, crs=gl_df.crs), axis=1)
        )
        all_cloud_stats = run_in_parallel(
            fun=functools.partial(
                compute_qc_stats,
                bands_name_map=bands_name_map,
                bands_qc_mask=bands_qc_mask,
                buffer_px=buffer_px
            ),
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

    cols_qc = ['fill_p', col_clouds, col_ndsi, col_albedo]
    gl_df = gl_df.merge(cloud_stats_df[['fp_img'] + cols_qc], on='fp_img')

    return gl_df


def select_best_images(
        gl_df: gpd.GeoDataFrame,
        fp_stats_all: Path,
        fp_stats_selected: Path,
        col_clouds: str,
        col_ndsi: str,
        col_albedo: str,
        max_n_imgs_per_g: int
) -> gpd.GeoDataFrame:
    """
    Select the best images based on the cloud coverage and the NDSI, plus the albedo as a tie-breaker.

    :param gl_df: the glacier dataframe containing already the image statistics
    :param fp_stats_all: the filepath to the csv file where to save the QC statistics for all images
    :param fp_stats_selected: the filepath to the csv file where to save the selected images
    :param col_clouds: the column name for the cloud and/or shadow coverage
    :param col_ndsi: the column name for the NDSI
    :param col_albedo: the column name for the albedo
    :param max_n_imgs_per_g: the maximum number of images to keep per glacier

    :return: a subset of the input dataframe containing the selected images and their statistics
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
        fp_gl_df_all: gpd.GeoDataFrame,
        out_rasters_dir: str | Path,
        raw_images_dir: str | Path = None,
        date_indices: tuple = (0, 8),
        date_format: str = '%Y%m%d',
        raw_fp_df: pd.DataFrame = None,
        buffer_px: int = 10,
        check_data_coverage: bool = True,
        no_data: int = C.NODATA,
        num_procs: int = 1,
        bands_name_map: dict = C.BANDS_NAME_MAP,
        bands_qc_mask: list = C.BANDS_QC_MASK,
        extra_geometries_dict: dict = None,
        extra_rasters_dict: dict = None,
        min_area: float = None,
        df_dates: pd.DataFrame = None,
        choose_best_auto: bool = False,
        max_cloud_f: float = None,
        min_fill_portion_scene: float = None,
        max_n_imgs_per_g: int = 1,
        compute_dem_features: bool = False
) -> None:
    """
    Prepare the rasters for all the glaciers in the given shapefile, with additional features.

    We need at least the following data sources:
    - a path to the shapefile containing the glacier outlines; the shapefile must contain the following columns:
        - entry_id: the glacier ID
        - geometry: the glacier outline
    - either:
        - a *directory* containing the raw images (e.g. Sentinel-2) for each glacier, structured in subdirectories as
        follows: raw_images_dir/glacier_id/image_name.tif. Note that the image name must contain the date of the image
        (e.g. 20190701.tif). The date is extracted from the image name using the date_indices and date_format
        parameters. The glacier_id is extracted from the directory name.

        or

        - a pandas *dataframe* containing the list of images for each glacier & date. The dataframe must contain the
        following columns:
            - entry_id: the glacier ID
            - fp_img: the filepath to the image
            - date: the date of the image in the format YYYY-MM-DD

        TODO: allow multiple images per date and glacier and stitch them together

    Each glacier-wide raster will contain:
    - the main data modality (e.g. optical data from Sentinel-2)
    - the current glacier outline as a binary mask, including a few masks with various buffers (see
    add_glacier_masks) around the main glacier which will be later used for evaluation statistics
    - a mask with all the glaciers in the image (where integer IDs are used)
    - the extra geometries (e.g. the debris cover) as binary masks, see the extra_geometries_dict parameter
    - the extra rasters (e.g. the DEM) which intersect the glacier outlines, see the extra_rasters_dict parameter
    - the features derived from the DEM, see the compute_dem_features parameter

    :param fp_gl_df_all: The filepath to the shapefile containing the glacier outlines.
    :param out_rasters_dir: The directory where to save the glacier-wide raster files.
    :param raw_images_dir: The directory containing the raw images (optional, see also *raw_fp_df*).
    :param date_indices: The range of where the date is located in the image filename (optional, see also *raw_fp_df*).
    :param date_format: The format of the date in the image filename (optional, see also *raw_fp_df*).
    :param raw_fp_df: The dataframe containing the list of images for each glacier (optional, if raw_images_dir is not
        given).
    :param buffer_px: The buffer in pixels to consider around the glacier outlines when cutting the raw images.
    :param check_data_coverage: If True, check if the raw image has a large enough spatial extent to cover the
        glacier + the buffer.
    :param no_data: The no data value to use for the output rasters.
    :param num_procs: The number of parallel processes to use.
    :param bands_name_map: A dict containing the bands we keep from the raw data (as keys) and their new names (values).
        If None, all the bands will be kept.
    :param bands_qc_mask: The bands to use for the QC mask (e.g. the cloud coverage).
    :param extra_geometries_dict: A dictionary containing the extra shapefiles to add to the glacier datasets as
        binary masks.
    :param extra_rasters_dict: A dictionary containing the extra rasters to add to the glacier datasets.
    :param min_area: The minimum glacier area to consider (only the rasters for these glaciers will be built but the
        main binary masks will contain all glaciers).
    :param df_dates: A dataframe containing the allowed dates for each glacier (if None, all the images will be used).
    :param choose_best_auto: If True, keep the best images based on the cloud coverage and the NDSI.
    :param max_cloud_f: The maximum cloud coverage allowed for each image.
    :param min_fill_portion_scene: The minimum data fill portion (i.e. not NAs) for the entire image.
    :param max_n_imgs_per_g: The maximum number of images to keep per glacier.
    :param compute_dem_features: If True, compute the features from the DEM (see the add_dem_features function).

    :return: None
    """

    print(f"Reading the glacier outlines in from {fp_gl_df_all}")
    gl_df_all = gpd.read_file(fp_gl_df_all)
    print(f"#glaciers = {len(gl_df_all)}")

    # check if the dataframes have the required ID columns
    assert 'entry_id' in gl_df_all.columns, "`entry_id` column not found in the glacier dataframe"
    if 'entry_id_i' not in gl_df_all.columns:
        print("Assigning an integer ID to each glacier, starting from 1, which will be used for the mask construction.")
        gl_df_all['entry_id_i'] = np.arange(len(gl_df_all)) + 1

    if min_area is not None:
        # keep the glaciers of interest in a different dataframe
        gl_df_sel = gl_df_all[gl_df_all.area_km2 >= min_area]
        print(f"#glaciers = {len(gl_df_sel)} after area filtering")
    else:
        gl_df_sel = gl_df_all

    assert (raw_images_dir is not None) or (raw_fp_df is not None), \
        "Either raw_images_dir or raw_fp_df must be given."

    if raw_fp_df is None:
        # get all the raw images and match them to the corresponding glaciers if the raw_fp_df is not given
        raw_images_dir = Path(raw_images_dir)
        assert raw_images_dir.exists(), f"raw_images_dir = {raw_images_dir} not found."
        fp_img_list_all = sorted(list(raw_images_dir.rglob('*.tif')))
        assert len(fp_img_list_all) > 0, f"No (tif) images found in {raw_images_dir}"

        raw_fp_df = pd.DataFrame({
            'entry_id': [fp.parent.name for fp in fp_img_list_all],
            'fp_img': fp_img_list_all,
            'date': [
                pd.to_datetime(fp.stem[date_indices[0]:date_indices[1]], format=date_format).strftime('%Y-%m-%d')
                for fp in fp_img_list_all
            ]
        })
    print(f"#total raw images = {len(raw_fp_df)}")

    if df_dates is not None:
        raw_fp_df = raw_fp_df.merge(df_dates, on=['entry_id', 'date'])
        print(f"#total raw images (after filtering by date) = {len(raw_fp_df)}")

    # add the filepaths to the selected glaciers and check the data coverage
    gid_list = set(gl_df_sel.entry_id)
    gl_df_sel = gl_df_sel.merge(raw_fp_df, on='entry_id', how='inner')  # we allow missing images

    def _show_crt_stats():
        print(f"\t#images = {len(gl_df_sel)}")
        print(f"\tavg. #images/glacier = {gl_df_sel.groupby('entry_id').count().fp_img.mean():.1f}")
        print(f"\t#glaciers with no data = {(n := len(gid_list - set(gl_df_sel.entry_id)))} "
              f"({n / len(gid_list) * 100:.1f}%)")

    print("After matching the images to the glaciers:")
    _show_crt_stats()

    # compute the cloud coverage statistics for each downloaded image if needed
    if choose_best_auto or max_cloud_f is not None or min_fill_portion_scene is not None:
        # specify the columns for the cloud coverage, the NDSI and the albedo stats (see the compute_qc_stats function)
        col_clouds = 'cloud_p_gl_b50m'
        col_ndsi = 'ndsi_avg_non_gl_b50m'
        col_albedo = 'albedo_avg_gl_b50m'

        fp_stats_all = Path(out_rasters_dir).parent / 'aux_data' / 'stats_all.csv'
        print(f"fp_stats_all = {fp_stats_all}")
        gl_df_sel = add_stats(
            gl_df=gl_df_sel,
            bands_name_map=bands_name_map,
            bands_qc_mask=bands_qc_mask,
            fp_stats=fp_stats_all,
            num_procs=num_procs,
            col_clouds=col_clouds,
            col_ndsi=col_ndsi,
            col_albedo=col_albedo,
            buffer_px=buffer_px,
        )

        # import the min fill portion threshold if given
        if min_fill_portion_scene is not None:
            gl_df_sel = gl_df_sel[gl_df_sel.fill_p >= min_fill_portion_scene]
            print(f"after NA filtering with min_fill_portion = {min_fill_portion_scene:.2f}")
            _show_crt_stats()

        # impose the max cloudiness threshold if given
        if max_cloud_f is not None:
            gl_df_sel = gl_df_sel[gl_df_sel[col_clouds] <= max_cloud_f]
            print(f"after cloud filtering with max_cloud_f = {max_cloud_f:.2f}")
            _show_crt_stats()

        # keep the best n images (based on cloud coverage and NDSI, plus albedo as tie-breaker) if needed
        if choose_best_auto:
            fp_qc_stats_all = Path(out_rasters_dir).parent / 'aux_data' / f'qc_stats_max_cloud_f_{max_cloud_f}.csv'
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
            _show_crt_stats()

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
    fn_img_out_list = [f"{x.stem[date_indices[0]:date_indices[1]]}.nc" for x in fp_img_list]
    fp_out_list = [Path(out_rasters_dir) / x / y for x, y in zip(gl_df_sel.entry_id, fn_img_out_list)]

    # build and export the rasters for each image
    run_in_parallel(
        fun=functools.partial(
            prep_glacier_dataset,
            gl_df=gl_df_all,
            extra_gdf_dict=extra_gdf_dict,
            bands_name_map=bands_name_map,
            bands_qc_mask=bands_qc_mask,
            buffer_px=buffer_px,
            check_data_coverage=check_data_coverage,
        ),
        fp_img=fp_img_list,
        fp_out=fp_out_list,
        entry_id=list(gl_df_sel.entry_id),
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
        fun=functools.partial(
            add_external_rasters,
            extra_rasters_bb_dict=extra_rasters_bb_dict,
            no_data=no_data
        ),
        fp_gl=fp_out_list,
        num_procs=num_procs,
        pbar=True
    )

    # derive features from the DEM if needed
    if compute_dem_features:
        run_in_parallel(
            fun=functools.partial(
                add_dem_features,
                no_data=no_data,
            ),
            fp_gl=fp_out_list,
            num_procs=num_procs,
            pbar=True
        )


if __name__ == "__main__":
    # the next settings apply to all datasets
    base_settings = dict(
        raw_images_dir=C.RAW_DATA_DIR,
        raw_fp_df=None,
        fp_gl_df_all=C.GLACIER_OUTLINES_FP,
        out_rasters_dir=C.DIR_GL_RASTERS,
        min_area=C.MIN_GLACIER_AREA,
        bands_name_map=C.BANDS_NAME_MAP,
        bands_qc_mask=C.BANDS_QC_MASK,
        no_data=C.NODATA,
        num_procs=C.NUM_PROCS,
        extra_geometries_dict=C.EXTRA_GEOMETRIES,
        extra_rasters_dict=C.EXTRA_RASTERS,
        compute_dem_features=True,
        buffer_px=(C.PATCH_RADIUS + int(round(50 / C.GSD))),
        min_fill_portion_scene=0.9
    )

    for k, v in {**C.EXTRA_GEOMETRIES, **C.EXTRA_RASTERS}.items():
        assert Path(v).exists(), f"{k} filepath: {v} does not exist"

    # some dataset specific settings
    specific_settings = {}
    if C.__name__ == 'S2_ALPS':
        specific_settings = dict(
            choose_best_auto=True,

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
            df_dates=dates_allowed
        )

    settings = base_settings.copy()
    settings.update(specific_settings)
    prepare_all_rasters(**settings)
