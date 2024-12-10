"""
    Script to assign for each glacier in the SGI2016 inventory the split in which it falls under the test fold.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # read the (processed) GLAMOS inventory (SGI2016)
    sgi_gdf = gpd.read_file('../data/outlines/sgi2016/SGI_2016_glaciers.shp')

    # for each glacier and data split, get the split in which it falls under the test fold
    fp_split_all = list(Path('../data/external/wd/s2_alps_plus/cv_split_outlines/').rglob('*.shp'))
    assert len(fp_split_all) == 5 * 3
    gdf_split_all = []
    for fp in fp_split_all:
        crt_gdf = gpd.read_file(fp)
        crt_gdf['split'] = fp.parent.name
        crt_gdf['fold'] = fp.stem
        gdf_split_all.append(crt_gdf)
    gdf_split_all = gpd.GeoDataFrame(pd.concat(gdf_split_all))
    print(gdf_split_all)

    sgi_to_closest_s2 = {}
    glamos_centroids = sgi_gdf.geometry.centroid.values
    _df = gdf_split_all.groupby('entry_id').first().set_crs(gdf_split_all.crs).to_crs(sgi_gdf.crs).reset_index()
    s2_centroids = _df.geometry.centroid.values
    for i in tqdm(range(len(sgi_gdf))):
        dists = s2_centroids.distance(glamos_centroids[i]) / 1e3
        i_closest = np.argmin(dists)
        sgi_to_closest_s2[sgi_gdf.iloc[i].sgi_id] = _df.iloc[i_closest].entry_id

    sgi_to_closest_s2_df = pd.Series(sgi_to_closest_s2).reset_index().rename(
        columns={'index': 'sgi_id', 0: 'entry_id'}
    )
    print(sgi_to_closest_s2_df)

    split_df = gdf_split_all.merge(sgi_to_closest_s2_df, on='entry_id')[['sgi_id', 'split', 'fold']]
    split_df = split_df.pivot(index='sgi_id', columns='split', values='fold').reset_index()
    split_df = split_df.rename(columns={'sgi_id': 'entry_id'})
    print(split_df)

    # save the GLAMOS inventory with the splits
    fp = Path('../data/external/wd/s2_sgi/cv_split_outlines/map_all_splits_all_folds.csv')
    fp.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(fp)
    print(f"SGI splits saved to {fp}")
