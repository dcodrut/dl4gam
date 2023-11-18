import geopandas as gpd
import pandas as pd
from pathlib import Path

# local imports
from utils.data_stats import compute_cloud_stats
from utils.general import run_in_parallel
from utils.data_prep import prep_raster
import config as C


def prepare_all_rasters(raw_images_dir, dems_dir, fp_gl_df_all, out_rasters_dir, extra_shp_dict=None):
    raw_images_dir = Path(raw_images_dir)
    assert raw_images_dir.exists(), f"raw_images_dir = {raw_images_dir} not found."
    subdir = raw_images_dir.name

    print(f"Reading the glacier outlines in from {fp_gl_df_all}")
    gl_df_all = gpd.read_file(fp_gl_df_all)
    print(f"#glaciers = {len(gl_df_all)}")

    # check if the dataframes have the required ID columns
    assert 'entry_id' in gl_df_all.columns
    assert 'entry_id_i' in gl_df_all.columns

    # keep the glaciers of interest in a different dataframe
    gl_df_sel = gl_df_all[gl_df_all.Area >= C.S2.MIN_GLACIER_AREA]
    # gl_df_sel = gl_df_sel.sort_values('Area', ascending=False).iloc[:1]
    print(f"#glaciers = {len(gl_df_sel)} after area filtering")

    # get all the raw images and match them to the corresponding glaciers
    fp_img_list_all = sorted(list(raw_images_dir.glob('**/*.tif')))
    raw_fp_df = pd.DataFrame({
        'entry_id': [fp.parent.name for fp in fp_img_list_all],
        'fp_img': fp_img_list_all
    })
    print(f"#total raw images = {len(raw_fp_df)}")

    # add the filepaths to the selected glaciers and check the data coverage
    gl_df_sel = gl_df_sel.merge(raw_fp_df, on='entry_id', how='inner')  # we allow missing images
    _ng = len(set(gl_df_sel.entry_id))
    _ni = len(gl_df_sel)
    print(f'after matching: '
          f'#raw images = {_ni}; #glaciers covered in the raw images = {_ng}; #images/glacier = {_ni / _ng:.1f}')

    # compute the cloud coverage statistics for each downloaded image if needed
    fp_cloud_stats = raw_images_dir.parent.parent / 'cloud_stats' / f"cloud_stats_{subdir}.csv"
    all_cloud_stats = run_in_parallel(
        fun=compute_cloud_stats,
        gl_sdf=[gl_df_sel[i:i + 1] for i in range(len(gl_df_sel))],
        num_cores=C.NUM_CORES,
        pbar=True
    )
    cloud_stats_df = pd.DataFrame.from_records(all_cloud_stats)
    cloud_stats_df = cloud_stats_df.sort_values(['entry_id', 'fp_img'])
    fp_cloud_stats.parent.mkdir(exist_ok=True)
    cloud_stats_df.to_csv(fp_cloud_stats, index=False)
    print(f"cloud stats exported to {fp_cloud_stats}")
    cloud_stats_df.fp_img = cloud_stats_df.fp_img.apply(lambda s: Path(s))

    # choose the least cloudy images and prepare the paths
    col_prc_clouds = 'cloud_p_v1_gl_only'
    gl_df_sel = gl_df_sel.merge(cloud_stats_df[['fp_img', col_prc_clouds]], on='fp_img')
    gl_df_sel = gl_df_sel.sort_values(col_prc_clouds).groupby('entry_id').first().reset_index()
    fp_img_list = [Path(x) for x in gl_df_sel.fp_img]

    # prepare the DEMs filepaths
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

    # read the extra shapefiles if given
    if extra_shp_dict is not None:
        extra_gdf_dict = {}
        for k, fp in extra_shp_dict.items():
            print(f"Reading the outlines {k} from {fp}")
            extra_gdf_dict[k] = gpd.read_file(fp)
    else:
        extra_gdf_dict = None

    # prepare the output paths
    fp_out_list = [Path(out_rasters_dir) / x.parent.name / x.with_suffix('.nc').name for x in fp_img_list]

    # build and export the rasters for each image
    run_in_parallel(
        fun=prep_raster,
        fp_img=fp_img_list,
        fp_dem=fp_dem_list,
        fp_out=fp_out_list,
        entry_id=list(gl_df_sel.entry_id),
        gl_df=gl_df_all,
        extra_gdf_dict=extra_gdf_dict,
        num_cores=C.NUM_CORES,
        pbar=True
    )


if __name__ == "__main__":
    # S2 data prep
    s2_extra_shp_dict = {
        'debris_scherler_2018': '../data/data_gmb/debris/scherler_2018/11_rgi60_CentralEurope_S2_DC_2015_2017_mode.shp',
        'debris_sgi_2016': '../data/data_gmb/glamos/inventory_sgi2016_r2020/SGI_2016_debriscover.shp',
    }
    for subdir in [C.S2.DIR_GL_RASTERS_2023]:
        subdir = Path(subdir).name
        settings = dict(
            raw_images_dir=f"../data/sat_data_downloader/external/download/s2/raw/{subdir}",
            dems_dir='../data/external/oggm/s2',
            fp_gl_df_all='../data/outlines/s2/rgi_format/c3s_gi_rgi11_s2_2015_v2/c3s_gi_rgi11_s2_2015_v2.shp',
            out_rasters_dir=f"../data/external/rasters/s2/{subdir}",
            extra_shp_dict=s2_extra_shp_dict
        )
        prepare_all_rasters(**settings)
