import functools
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.ops
import xarray as xr

# local imports
from config import C, S2_PS, PS
from utils.data_prep import prep_glacier_dataset, add_external_raster, add_dem_features
from utils.data_stats import compute_qc_stats
from utils.general import run_in_parallel


def prepare_all_rasters(raw_images_dir, dems_dir, fp_gl_df_all, out_rasters_dir, bands_to_keep, buffer_px, no_data,
                        num_cores, extra_shp_dict=None, extra_rasters_dict=None, min_area=None, choose_best_auto=False,
                        max_cloud_f=None, max_n_imgs_per_g=1, df_dates=None, compute_dem_features=False):
    raw_images_dir = Path(raw_images_dir)
    assert raw_images_dir.exists(), f"raw_images_dir = {raw_images_dir} not found."

    print(f"Reading the glacier outlines in from {fp_gl_df_all}")
    gl_df_all = gpd.read_file(fp_gl_df_all)
    print(f"#glaciers = {len(gl_df_all)}")

    # check if the dataframes have the required ID columns
    assert 'entry_id' in gl_df_all.columns
    assert 'entry_id_i' in gl_df_all.columns

    if min_area is not None:
        # keep the glaciers of interest in a different dataframe
        gl_df_sel = gl_df_all[gl_df_all.Area >= min_area]
        # gl_df_sel = gl_df_sel.sort_values('Area', ascending=False).iloc[:1]
        print(f"#glaciers = {len(gl_df_sel)} after area filtering")
    else:
        gl_df_sel = gl_df_all

    # get all the raw images and match them to the corresponding glaciers
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
    print(f"avg. #images/glacier = {gl_df_sel.groupby('entry_id').count().gid.mean():.1f}")
    print(f"#glaciers with no data = {(n := len(gid_list - set(gl_df_sel.entry_id)))} ({n / len(gid_list) * 100:.1f}%)")

    # compute the cloud coverage statistics for each downloaded image if needed
    if choose_best_auto or max_cloud_f is not None:
        fp_cloud_stats = Path(out_rasters_dir).parent / 'aux_data' / 'cloud_stats.csv'
        if fp_cloud_stats.exists():
            print(f"Reading the cloud stats from {fp_cloud_stats}")
            cloud_stats_df = pd.read_csv(fp_cloud_stats)
        else:
            gl_sdf_list = list(
                gl_df_sel.apply(lambda r: gpd.GeoDataFrame(pd.DataFrame(r).T, crs=gl_df_sel.crs), axis=1)
            )
            all_cloud_stats = run_in_parallel(
                fun=functools.partial(compute_qc_stats, include_shadows=False),
                gl_sdf=gl_sdf_list,
                num_cores=num_cores,
                pbar=True
            )
            cloud_stats_df = pd.DataFrame.from_records(all_cloud_stats)
            cloud_stats_df = cloud_stats_df.sort_values(['entry_id', 'fp_img'])
            fp_cloud_stats.parent.mkdir(exist_ok=True, parents=True)
            cloud_stats_df.to_csv(fp_cloud_stats, index=False)
            print(f"cloud stats exported to {fp_cloud_stats}")
        cloud_stats_df.fp_img = cloud_stats_df.fp_img.apply(lambda s: Path(s))

        # choose the least cloudy images and prepare the paths
        col_clouds = 'cloud_p_v1_gl_only'
        col_ndsi = 'ndsi_avg_scene_v1'
        col_albedo = 'albedo_avg_gl_v1'
        cols_qc = [col_clouds, col_ndsi, col_albedo]
        gl_df_sel = gl_df_sel.merge(cloud_stats_df[['fp_img'] + cols_qc], on='fp_img')

        # impose the max cloudiness threshold if given (this includes missing pixels)
        if max_cloud_f is not None:
            gl_df_sel = gl_df_sel[gl_df_sel[col_clouds] <= max_cloud_f]
            print(f"after cloud filtering with max_cloud_f = {max_cloud_f:.2f}")
            print(f"\t#images = {len(gl_df_sel)}")
            print(f"\tavg. #images/glacier = {gl_df_sel.groupby('entry_id').count().gid.mean():.1f}")
            print(f"\t#glaciers with no data = {(n := len(gid_list - set(gl_df_sel.entry_id)))} "
                  f"({n / len(gid_list) * 100:.1f}%)")

        # keep the best n images (based on cloud coverage and NDSI)
        if choose_best_auto:
            # give an index to each image based on the cloud coverage and the NDSI
            for c in [col_clouds, col_ndsi]:
                df_idx = gl_df_sel.groupby('entry_id', sort=True, group_keys=False).apply(
                    lambda ssdf: pd.DataFrame(
                        {
                            'entry_id': ssdf.entry_id,
                            'date': ssdf.date,
                            f"{c}_s": np.searchsorted(ssdf[c].round(2).sort_values(), ssdf[c].round(2)),
                        }
                    )
                )
                gl_df_sel = gl_df_sel.merge(df_idx, on=['entry_id', 'date'], validate='one_to_one')

            # compute the combined score based on the cloud coverage and the NDSI
            gl_df_sel['qc_score'] = (gl_df_sel[f"{col_clouds}_s"] + gl_df_sel[f"{col_ndsi}_s"]) / 2
            # export the scores
            fp_sel = Path(out_rasters_dir).parent / 'aux_data' / 'qc_stats.csv'
            pd.DataFrame(gl_df_sel.drop(columns='geometry')).to_csv(fp_sel, index=False)
            print(f"Selected images exported to {fp_sel}")

            # keep the best (use albedo as tie-breaker)
            gl_df_sel = gl_df_sel.sort_values(['qc_score', col_albedo])
            gl_df_sel = gl_df_sel.groupby('entry_id').head(max_n_imgs_per_g).reset_index()
            print(f"after keeping the best {max_n_imgs_per_g} images per glacier:")
            print(f"\t#images = {len(gl_df_sel)}")
            print(f"\tavg. #images/glacier = {gl_df_sel.groupby('entry_id').count().gid.mean():.1f}")
            print(f"\t#glaciers with no data = {(n := len(gid_list - set(gl_df_sel.entry_id)))} "
                  f"({n / len(gid_list) * 100:.1f}%)")

            # export the selected dataframe
            fp_sel = Path(out_rasters_dir).parent / 'aux_data' / 'selected_images.csv'
            gl_df_sel = gl_df_sel.sort_values(['entry_id', 'date'])
            pd.DataFrame(gl_df_sel.drop(columns='geometry')).to_csv(fp_sel, index=False)
            print(f"Selected images exported to {fp_sel}")

    # prepare the DEMs filepaths
    if dems_dir is not None:
        fp_dem_list_all = list(Path(dems_dir).glob('**/dem.tif'))
        dem_fp_df = pd.DataFrame({
            'RGIId': [fp.parent.name for fp in fp_dem_list_all],
            'fp_dem': fp_dem_list_all
        })
        print(f"#DEMs = {len(dem_fp_df)}")
        rgi_ids_without_dems = set(gl_df_sel.RGIId) - set(dem_fp_df.RGIId)
        assert len(rgi_ids_without_dems) == 0, f"No DEMs found for {rgi_ids_without_dems}"
        gl_df_sel = gl_df_sel.merge(dem_fp_df, on='RGIId')
        fp_dem_list = list(gl_df_sel.fp_dem)
    else:
        fp_dem_list = None

    # read the extra shapefiles if given
    if extra_shp_dict is not None:
        extra_gdf_dict = {}
        for k, fp in extra_shp_dict.items():
            print(f"Reading the outlines {k} from {fp}")
            extra_gdf_dict[k] = gpd.read_file(fp)
    else:
        extra_gdf_dict = None

    # prepare the output paths
    fp_img_list = [Path(x) for x in gl_df_sel.fp_img]
    fp_out_list = [Path(out_rasters_dir) / x.parent.name / x.with_suffix('.nc').name for x in fp_img_list]

    # build and export the rasters for each image
    run_in_parallel(
        fun=prep_glacier_dataset,
        fp_img=fp_img_list,
        fp_dem=fp_dem_list,
        fp_out=fp_out_list,
        entry_id=list(gl_df_sel.entry_id),
        gl_df=gl_df_all,
        extra_gdf_dict=extra_gdf_dict,
        bands_to_keep=bands_to_keep,
        buffer_px=buffer_px,
        no_data=no_data,
        num_cores=num_cores,
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
        fun=add_external_raster,
        fp_gl=fp_out_list,
        extra_rasters_bb_dict=extra_rasters_bb_dict,
        num_cores=num_cores,
        no_data=no_data,
        pbar=True
    )

    # derive features from the DEM if needed
    if add_dem_features:
        run_in_parallel(
            fun=add_dem_features,
            fp_gl=fp_out_list,
            num_cores=num_cores,
            no_data=no_data,
            pbar=True
        )


if __name__ == "__main__":
    # specify which other outlines to use to build masks and add them to the final rasters
    extra_shp_dict = C.DEBRIS_OUTLINES_FP

    # the next settings apply to all datasets
    base_settings = dict(
        raw_images_dir=C.RAW_DATA_DIR,
        dems_dir=None,
        fp_gl_df_all=C.GLACIER_OUTLINES_FP,
        out_rasters_dir=C.DIR_GL_INVENTORY,
        min_area=C.MIN_GLACIER_AREA,
        bands_to_keep=C.BANDS,
        no_data=C.NODATA,
        num_cores=C.NUM_CORES,
        extra_shp_dict=extra_shp_dict,
        extra_rasters_dict={
            'dem': C.DEMS_DIR,
            'dhdt': C.DHDT_DIR
        },
        compute_dem_features=True
    )

    # some dataset specific settings
    specific_settings = {}
    if C.__name__ == 'S2':
        specific_settings = dict(
            choose_least_cloudy=True,
            buffer_px=C.PATCH_RADIUS,
        )
    elif C.__name__ == 'S2_PLUS':
        if C.CSV_DATES_ALLOWED is not None:
            dates_allowed = pd.read_csv(C.CSV_DATES_ALLOWED, converters={'entry_id': str})
            max_cloud_f = None
            choose_best_auto = False
        else:
            dates_allowed = None
            max_cloud_f = 0.25
            choose_best_auto = True
        specific_settings = dict(
            buffer_px=C.PATCH_RADIUS,
            df_dates=dates_allowed,
            max_cloud_f=max_cloud_f,
            choose_best_auto=choose_best_auto
        )
    elif C.__name__ == 'S2_GLAMOS':
        print(f"Reading the allowed dates csv from {C.CSV_DATES_ALLOWED}")
        dates_allowed = pd.read_csv(C.CSV_DATES_ALLOWED, converters={'entry_id': str})
        specific_settings = dict(
            choose_least_cloudy=True,
            buffer_px=C.PATCH_RADIUS,
            df_dates=dates_allowed
        )
    elif C.__name__ == 'PS':
        specific_settings = dict(
            choose_least_cloudy=False,
            # increase the buffer even if it's not needed to have the same spatial extend as S2 data
            buffer_px=int(PS.PATCH_RADIUS * (S2_PS.PATCH_RADIUS * 10) / (PS.PATCH_RADIUS * 3)),
        )
    elif C.__name__ == 'S2_PS':
        # S2 data prep that matches the Planet data
        # in this case multiple rasters per glacier were initially produced, then manually chose the best, as for Planet
        # the dates for the final rasters were saved in the provided csv file
        print(f"Reading the final dates csv from {C.CSV_FINAL_DATES}")
        final_dates = pd.read_csv(C.CSV_FINAL_DATES, converters={'entry_id': str})
        specific_settings = dict(
            buffer_px=C.PATCH_RADIUS,
            df_dates=final_dates
        )

    settings = base_settings.copy()
    settings.update(specific_settings)
    prepare_all_rasters(**settings)
