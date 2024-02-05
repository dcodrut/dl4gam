from pathlib import Path
import xarray as xr
import numpy as np
import shapely
import shapely.ops
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import gc

# local imports
import config as C

if __name__ == "__main__":
    input_dir = Path('../data/data_gmb/data_gl_seg/external/planet/inv/raw/')
    output_dir = Path('../data/data_gmb/data_gl_seg/external/planet/raw_processed/inv/')

    # prepare the glacier outlines and their buffers
    outlines_fp = '../data/data_gmb/outlines_2015/c3s_gi_rgi11_s2_2015_v2/c3s_gi_rgi11_s2_2015_v2.shp'
    s2_df = gpd.read_file(outlines_fp)
    s2_df['year'] = s2_df.Date.apply(lambda s: pd.to_datetime(s).year)
    s2_df['month'] = s2_df.Date.apply(lambda s: pd.to_datetime(s).month)
    s2_df['day'] = s2_df.Date.apply(lambda s: pd.to_datetime(s).day)
    buffer_m = 1280
    s2_df_buffer = s2_df[s2_df.AREA_KM2 >= 0.1].copy()
    print(f"without any buffer; total area = {s2_df_buffer.area.sum() / 1e6:.3f} km2")

    # draw a rectangle box with a buffer around each glacier
    rectangle_buffers = s2_df_buffer.geometry.apply(
        lambda g: shapely.geometry.box(*shapely.geometry.box(*g.bounds).buffer(buffer_m).bounds))
    print(f"rectangle buffer_m = {buffer_m}; total area = {rectangle_buffers.area.sum() / 1e6:.3f} km2")
    print(f"rectangle buffer_m = {buffer_m}; "
          f"total area after reduction = {shapely.ops.unary_union(rectangle_buffers).area / 1e6:.3f} km2")

    # draw a simple buffer around each glacier
    simple_buffers = s2_df_buffer.buffer(buffer_m)
    print(f"simple buffer_m = {buffer_m}; total area = {simple_buffers.area.sum() / 1e6:.3f} km2")
    print(f"simple buffer_m = {buffer_m};"
          f" total area after reduction = {shapely.ops.unary_union(simple_buffers).area / 1e6:.3f} km2")

    # set the desired buffers
    s2_df_buffer.geometry = rectangle_buffers
    s2_df_buffer = s2_df_buffer.to_crs(epsg=4326)
    print(s2_df_buffer)

    # prepare the Planet image path for each glacier
    shp_list = list(
        Path('../data/data_gmb/data_gl_seg/external/planet/inv/outlines_boxes_buffer_1280m/').glob('**/*.shp'))
    gdf_boxes = []
    for fp in shp_list:
        crt_gdf = gpd.read_file(fp)
        crt_gdf['fn'] = fp.stem
        gdf_boxes.append(crt_gdf)
    gdf_boxes = gpd.GeoDataFrame(pd.concat(gdf_boxes))
    gdf_boxes = gdf_boxes.sort_values(['date', 's2_tile', 'group', 'id'])
    print(gdf_boxes)

    gl_to_box = {}
    for _, row in gdf_boxes.iterrows():
        # get the glaciers (with their buffers) for the current date & tile
        s2_sdf_buffer = s2_df_buffer[(s2_df_buffer.Date == row.date) & (s2_df_buffer.Tile_Name == row.s2_tile)]

        # keep only those contained by the current boxes
        s2_sdf_buffer = s2_sdf_buffer[row.geometry.buffer(1e-5).contains(s2_sdf_buffer.geometry)]

        for gid in s2_sdf_buffer.GLACIER_NR:
            gl_to_box[gid] = row.fn
    gl_to_box_df = pd.Series(gl_to_box).reset_index().rename(columns={'index': 'gl_num', 0: 'fn'})
    gl_to_box_df = gl_to_box_df.sort_values('gl_num')
    gl_to_box_df.to_csv('../data/data_gmb/data_gl_seg/external/planet/inv/gl_to_box_df.csv')

    # manually compose a few images which were not downloaded as composite
    if False:
        import rioxarray
        import rioxarray.merge

        fp_list = list(Path('../data/data_gmb/data_gl_seg/external/planet/inv/raw/').glob('**/*.tif'))
        # remove the Unusable Data Masks
        fp_list = list(filter(lambda x: 'udm2' not in x.name, fp_list))
        for suf in ['.tif', '_udm2.tif']:
            x = sorted(list(filter(lambda fp: 'boxes_2015-08-29_tile_32TLR_items_1_' in str(fp), fp_list)))
            x = [y.parent / (y.stem + suf) for y in x]
            d1 = xr.open_dataarray(x[0], cache=False)
            d2 = xr.open_dataarray(x[1], cache=False)
            d3 = xr.open_dataarray(x[2], cache=False)
            d123 = rioxarray.merge.merge_arrays([d2, d1, d3])
            fp_out = x[0].parent / ('composite' + suf)
            NODATA = -9999
            d123 = d123.fillna(NODATA).astype(np.int16)
            d123.attrs['_FillValue'] = NODATA
            d123.rio.to_raster(fp_out)
            print(f"data exported to {fp_out}")

            # delete the original files
            for xx in x:
                xx.unlink()

    # add the corresponding raw filenames to each glacier
    fp_list = list(input_dir.glob('**/*.tif'))
    # remove the Unusable Data Masks
    fp_list = list(filter(lambda x: 'udm2' not in x.name, fp_list))
    print(fp_list[:3])
    fp_list_merge = []
    for _, r in gl_to_box_df.iterrows():
        x = list(filter(lambda f: (r.fn + '_') in str(f), fp_list))
        assert len(x) == 1
        fp_list_merge.append(x[0])
    gl_to_box_df['fp'] = fp_list_merge
    print(gl_to_box_df)
    s2_df_buffer_box = s2_df_buffer.rename(columns={'GLACIER_NR': 'gl_num'}).merge(gl_to_box_df)
    print(s2_df_buffer_box)

    # build a pandas dataframe with the corresponding image data for each glacier
    # in most of the cases, there should be one single date; treat the exceptions manually
    dates_shp_dir = Path('../data/data_gmb/data_gl_seg/external/planet/inv/dates_mixed')
    boxes_with_mixed_dates = [
        'boxes_2015-08-29_tile_32TLR_items_1',
        'boxes_2015-08-29_tile_32TMS_items_14',
        'boxes_2015-08-29_tile_32TMS_items_19',
        'boxes_2016-09-29_tile_32TPS_items_6-10'
    ]
    gl_to_date = {}
    for raw_fp in tqdm(sorted(set(s2_df_buffer_box.fp))):
        # get all the glaciers from the current box
        box_name = Path(str(raw_fp).split('_pss')[0]).name
        gid_list = list(s2_df_buffer_box[s2_df_buffer_box.fn == box_name].gl_num)

        # get the acquisition dates using the metadata file names
        meta_files = list(filter(lambda s: 'composite' not in s.name, list(raw_fp.parent.glob('*_metadata.json'))))
        dates = sorted(list(set([fp.stem[:8] for fp in meta_files])))

        if box_name in boxes_with_mixed_dates:
            _fp = dates_shp_dir / box_name / f"{dates[0]}.shp"
            gid_list_first_date = list(gpd.read_file(_fp).GLACIER_NR)
            assert len(set(gid_list_first_date) - set(gid_list)) == 0
            gl_to_date.update({gid: dates[0] for gid in gid_list_first_date})
            gl_to_date.update({gid: dates[1] for gid in set(gid_list) - set(gid_list_first_date)})
        else:
            assert len(dates) == 1
            gl_to_date.update({gid: dates[0] for gid in gid_list})
    gl_to_date_df = pd.Series(gl_to_date).reset_index().rename(columns={'index': 'GLACIER_NR', 0: 'date_ps'})
    gl_to_date_df = gl_to_date_df.sort_values('GLACIER_NR')
    assert len(gl_to_date_df) == len(gl_to_box_df)
    gl_to_date_df.to_csv('../data/data_gmb/data_gl_seg/external/planet/inv/dates.csv', index=False)

    # cut the image for each glacier
    for box_name in tqdm(sorted(set(s2_df_buffer_box.fn))):
        raw_fp = s2_df_buffer_box[s2_df_buffer_box.fn == box_name].fp.iloc[0]
        img_planet = xr.open_dataarray(raw_fp, cache=False).fillna(C.PS.NODATA).astype(np.int16)
        img_planet.attrs['_FillValue'] = C.PS.NODATA

        # read the corresponding mask and keep only the necessary bands
        mask_name = (raw_fp.stem + '_udm2.tif') if 'AnalyticMS_SR' not in raw_fp.stem \
            else raw_fp.name.replace('AnalyticMS_SR_harmonized', 'udm2')
        mask_planet = xr.open_dataarray(raw_fp.parent / mask_name, cache=False)
        idx_bands = [mask_planet.attrs['long_name'].index(x) for x in ['cloud', 'shadow']]
        mask_planet = mask_planet.isel(band=idx_bands).fillna(C.PS.NODATA).astype(np.int16)
        mask_planet.attrs['_FillValue'] = C.PS.NODATA

        # add the masks to the image
        img_planet_with_mask = xr.concat([img_planet, mask_planet], dim='band')
        img_planet.close()
        mask_planet.close()

        # reassign the coordinates and add their name
        img_planet_with_mask = img_planet_with_mask.assign_coords(band=np.arange(len(img_planet_with_mask.band)) + 1)
        img_planet_with_mask.attrs['long_name'] = ['B', 'G', 'R', 'NIR', 'cloud', 'shadow']

        s2_sdf_buffer_box = s2_df_buffer_box[s2_df_buffer_box.fp == raw_fp].to_crs(img_planet_with_mask.rio.crs)
        for j in tqdm(range(len(s2_sdf_buffer_box)), desc=box_name):
            r = s2_sdf_buffer_box.iloc[j]
            date = gl_to_date[r.gl_num]
            fp_out = output_dir / f"{r.gl_num:04d}" / f"{date}.tif"
            if fp_out.exists():
                continue
            img_crt_g = img_planet_with_mask.rio.clip([r.geometry])
            img_crt_g.attrs['fn'] = date
            fp_out.parent.mkdir(parents=True, exist_ok=True)
            img_crt_g.rio.to_raster(fp_out)

            gc.collect()
