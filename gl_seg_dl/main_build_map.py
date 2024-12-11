import webbrowser
from pathlib import Path

import geopandas as gpd
import leafmap.foliumap as leafmap
import folium
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # read the inventory
    gdf_inv = gpd.read_file('../data/outlines/paul_et_al_2020/c3s_gi_rgi11_s2_2015_v2.shp')

    polys_root_dir = Path('../data/external/_experiments/s2_alps_plus/unet/gdf_all_splits/seed_all/version_0/s2_alps_plus')
    gdf_pred_0 = gpd.read_file(polys_root_dir / 'inv/inv_pred.shp')
    gdf_pred_1 = gpd.read_file(polys_root_dir / '2023/2023_pred.shp')

    # add the area in km2
    gdf_inv['area_km2'] = gdf_inv.area / 1e6
    gdf_pred_0['area_km2'] = gdf_pred_0.area / 1e6
    gdf_pred_1['area_km2'] = gdf_pred_1.area / 1e6

    gdf_pred_0 = gdf_pred_0.to_crs(epsg=4326)
    gdf_pred_1 = gdf_pred_1.to_crs(epsg=4326)

    aletsch_loc = (46.50410, 8.03522)

    m = leafmap.Map(height=600, center=aletsch_loc, zoom=12, scroll_wheel_zoom=True)

    base_maps = [
        'OpenTopoMap',
        'SATELLITE',
        'TERRAIN',
        'HYBRID',
        'Esri.WorldImagery',
        'Esri.WorldShadedRelief',
        'SwissFederalGeoportal.JourneyThroughTime',
        'SwissFederalGeoportal.SWISSIMAGE'
    ]
    for c in base_maps:
        m.add_basemap(c, show=(c == 'Esri.WorldImagery'))

    # add the shapefile
    with plt.style.context('tableau-colorblind10'):
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    style_inv = {
        "stroke": True,
        "color": color_list[2],
        "weight": 2.0,
        "opacity": 1,
        "fill": True,
        "fillColor": color_list[2],
        "fillOpacity": 0.1,
    }
    style_pred_0 = style_inv.copy()
    style_pred_0['color'] = color_list[0]
    style_pred_0['fillColor'] = color_list[0]

    m.add_gdf(gdf_pred_0, layer_name='2015/16/17', style=style_pred_0)
    style_pred1 = style_pred_0.copy()
    style_pred1['color'] = color_list[1]
    style_pred1['fillColor'] = color_list[1]

    m.add_gdf(gdf_pred_1, layer_name='2023', style=style_pred1)

    # add a tooltip for each glacier centroid
    for idx, row in gdf_pred_0.iterrows():
        m.add_marker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            popup=f'<div style="width: 200px; font-size: 16px;" <b>Area:</b> {row.area_km2:.1f} km<sup>2</sup></div>',
            tooltip=row.entry_id,
            # icon='fa-snowflake'
        )

    folium.LayerControl(sortLayers=True).add_to(m)

    out_fp = Path('../data/maps/index.html')
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.unlink(missing_ok=True)
    m.to_html(str(out_fp))

    webbrowser.open(str(out_fp))

