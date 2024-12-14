import webbrowser
from pathlib import Path

import folium
import geopandas as gpd
import leafmap.foliumap as leafmap
import matplotlib.pyplot as plt
import pandas as pd
from folium.plugins import MarkerCluster

if __name__ == "__main__":
    # read the inventory
    gdf_inv = gpd.read_file('../data/outlines/paul_et_al_2020/c3s_gi_rgi11_s2_2015_v2.shp')

    # keep only the glaciers covered in our dataset
    df_final_dates = pd.read_csv('../data/inv_images_qc/final_dates.csv')
    gdf_inv = gdf_inv[gdf_inv.entry_id.isin(df_final_dates.entry_id)]

    # read the predictions
    polys_root_dir = Path(
        '../data/external/_experiments/s2_alps_plus/unet/gdf_all_splits/seed_all/version_0/s2_alps_plus')
    gdf_pred_0 = gpd.read_file(polys_root_dir / 'inv/inv_pred.shp')
    gdf_pred_1 = gpd.read_file(polys_root_dir / '2023/2023_pred.shp')

    # read the lower and upper bounds
    gdf_pred_lb_0 = gpd.read_file(polys_root_dir / 'inv/inv_pred_low.shp')
    gdf_pred_ub_0 = gpd.read_file(polys_root_dir / 'inv/inv_pred_high.shp')
    gdf_pred_lb_1 = gpd.read_file(polys_root_dir / '2023/2023_pred_low.shp')
    gdf_pred_ub_1 = gpd.read_file(polys_root_dir / '2023/2023_pred_high.shp')

    # add the area in km2 and then convert to WGS84
    for df in [gdf_inv, gdf_pred_0, gdf_pred_1, gdf_pred_lb_0, gdf_pred_ub_0, gdf_pred_lb_1, gdf_pred_ub_1]:
        df['area_km2'] = (df.to_crs(epsg=32632).area / 1e6).round(4)
        df.drop(columns=['label'], errors='ignore', inplace=True)
        df.to_crs(epsg=4326, inplace=True)

    # build the confidence intervals
    gdf_pred_0_unc = gdf_pred_lb_0.merge(gdf_pred_ub_0, on='entry_id', suffixes=('_lb', '_ub'))
    gdf_pred_0_unc['geometry'] = gdf_pred_0_unc.geometry_ub.difference(gdf_pred_0_unc.geometry_lb)
    gdf_pred_0_unc = gdf_pred_0_unc.drop(columns=['geometry_lb', 'geometry_ub'])
    gdf_pred_1_unc = gdf_pred_lb_1.merge(gdf_pred_ub_1, on='entry_id', suffixes=('_lb', '_ub'))
    gdf_pred_1_unc['geometry'] = gdf_pred_1_unc.geometry_ub.difference(gdf_pred_1_unc.geometry_lb)
    gdf_pred_1_unc = gdf_pred_1_unc.drop(columns=['geometry_lb', 'geometry_ub'])

    # read the processed rates
    df_rates = pd.read_csv('../data/external/_experiments/s2_alps_plus/unet/stats_rates_s2_alps_plus_version_0.csv')

    # make sure all the dataframes have the same glaciers
    covered_glaciers = set(gdf_inv.entry_id)
    assert covered_glaciers == set(gdf_pred_0.entry_id)
    assert covered_glaciers == set(gdf_pred_1.entry_id)
    assert covered_glaciers == set(gdf_pred_0_unc.entry_id)
    assert covered_glaciers == set(gdf_pred_1_unc.entry_id)
    assert covered_glaciers.issubset(set(df_rates.entry_id))  # the rates also include the extrapolations

    aletsch_loc = (46.50410, 8.03522)
    m = leafmap.Map(height=600, center=aletsch_loc, zoom=12, scroll_wheel_zoom=True, draw_control=False)

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

    # change each tile layer to overlay=False
    for k, v in m._children.items():
        if isinstance(v, folium.raster_layers.TileLayer):
            v.overlay = False

    with plt.style.context('tableau-colorblind10'):
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # add the inventory
    style_inv = {
        "stroke": True,
        "color": color_list[3],
        "weight": 3.0,
        "opacity": 1,
        "fill": True,
        "fillColor": color_list[3],
        "fillOpacity": 0.2,

    }
    gdf_inv.date_inv = gdf_inv.date_inv.apply(lambda x: x.strftime('%Y-%m-%d'))

    # add the predictions and the uncertainty for 2015/16/17
    tooltip = folium.GeoJsonTooltip(
        fields=list(gdf_inv.columns)[:-3],
        style=("background-color: white; color: black; font-size: 16px;")
    )
    folium.GeoJson(
        gdf_inv,
        name='Inventory 2015/16/17 (Paul et al. 2020)',
        style_function=lambda x: style_inv,
        highlight_function=lambda feat: {
            "weight": 4,
            "fillOpacity": 0.5
        },
        tooltip=tooltip,
    ).add_to(m)

    style_pred_0 = style_inv.copy()
    style_pred_0['color'] = color_list[0]
    style_pred_0['fillColor'] = color_list[0]
    m.add_gdf(gdf_pred_0, layer_name='DL4GAM prediction 2015/16/17', style=style_pred_0, zoom_to_layer=False)

    style_pred_unc = style_pred_0.copy()
    # hide the border & increase the fill opacity
    style_pred_unc['opacity'] = 0.0
    style_pred_unc['fillOpacity'] = 0.4
    m.add_gdf(
        gdf_pred_0_unc[['entry_id', 'geometry']],
        layer_name='DL4GAM 1σ-uncertainty 2015/16/17',
        style=style_pred_unc,
        zoom_to_layer=False,
        show=False
    )

    # add the predictions and the uncertainty for 2023
    style_pred1 = style_pred_0.copy()
    style_pred1['color'] = color_list[1]
    style_pred1['fillColor'] = color_list[1]
    m.add_gdf(gdf_pred_1, layer_name='DL4GAM prediction 2023', style=style_pred1, zoom_to_layer=False)

    style_pred_unc1 = style_pred1.copy()
    style_pred_unc1['opacity'] = 0.0
    style_pred_unc1['fillOpacity'] = 0.4
    m.add_gdf(
        gdf_pred_1_unc,
        layer_name='DL4GAM 1σ-uncertainty 2023',
        style=style_pred_unc1,
        zoom_to_layer=False,
        show=False
    )

    # add a tooltip for each glacier centroid
    marker_cluster = MarkerCluster(name="Information markers").add_to(m)
    for entry_id in gdf_pred_0.entry_id:
        r_inv = gdf_inv[gdf_inv.entry_id == entry_id].iloc[0]
        r = df_rates[df_rates.entry_id == entry_id].iloc[0]
        desc = {
            f"Area {r.year_t0} (inventory):": f"{r.area_inv:.4f} km²",
            f'Area {r.year_t0} (DL4GAM):': f"{r.area_t0:.4f} ± {r.area_t0_std:.4f} km²",
            f'Area {r.year_t1} (DL4GAM):': f"{r.area_t1:.4f} ± {r.area_t1_std:.4f} km²",
            f'Annual area change rate:': f"{r.area_rate:.4f} ± {r.area_rate_std:.4f} km² y⁻¹",
            ';'.join(['&nbsp'] * 42): f"({r.area_rate_prc * 100:.2f} ± {r.area_rate_std / r.area_t0 * 100:.2f} % y⁻¹)",
        }
        folium.Marker(
            location=[r_inv.LAT, r_inv.LON],
            popup=f'<div style="font-size: 16px; white-space: nowrap;">'
                  f'<b>Summary for glacier {entry_id.replace("g_", "")}:</b><br><br>' +
                  ''.join([f'<b>{k}</b> {v}<br>' for k, v in desc.items()]) + '</div>',
            tooltip=f'<div style="font-size: 16px;">Click for details.</div>',
            icon=folium.Icon(icon='fa-snowflake', prefix='fa')
        ).add_to(marker_cluster)

    # Add layer controls
    folium.LayerControl(position='topright', collapsed=False, autoZIndex=True, draggable=True).add_to(m)

    out_fp = Path('../data/maps/index.html')
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.unlink(missing_ok=True)
    m.to_html(str(out_fp))

    webbrowser.open(str(out_fp))
