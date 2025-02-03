from functools import partial
from pathlib import Path

import geopandas as gpd
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib.ticker import FormatStrFormatter
from matplotlib_scalebar.scalebar import ScaleBar

# local imports
from config import C
from utils.general import run_in_parallel
from utils.viz_utils import contrast_stretch

with plt.style.context(("tableau-colorblind10")):
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_glacier(
        fp_raster,
        gl_df,
        gl_df_pred,
        df_results,
        plot_dir,
        fig_w_px=1920,
        dpi=150,
        line_thickness=1,
        fontsize=12
):
    nc = xr.open_dataset(fp_raster, decode_coords='all')
    img_date = pd.to_datetime(fp_raster.name[:8])

    # set the width ratios (the 3 images + the colorbar) and prepare the figure
    width_ratios = [1, 1, 1, 0.035]

    # resample s.t. the image width fits the desired figure width
    img_w_px = int((fig_w_px - width_ratios[-1] * fig_w_px) / 3)
    scale_factor = img_w_px / len(nc.x)
    img_h_px = int(len(nc.y) * scale_factor)
    nc = nc.rio.reproject(dst_crs=nc.rio.crs, shape=(img_h_px, img_w_px), resampling=rasterio.enums.Resampling.bilinear)
    extent = [float(x) for x in [nc.x.min(), nc.x.max(), nc.y.min(), nc.y.max()]]

    # prepare the figure
    fig_w = fig_w_px / dpi
    h_title_px = img_w_px * 0.1
    fig_h = (img_h_px + h_title_px) / dpi
    fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h), dpi=dpi, width_ratios=width_ratios, layout='compressed')

    # Subplot 1) plot the RGB image with the glacier outlines
    ax = axes[0]
    band_names = nc.band_data.long_name
    img = nc.band_data.isel(band=[band_names.index(b) for b in ('B4', 'B3', 'B2')]).transpose('y', 'x', 'band').values
    img = contrast_stretch(img=img, q_lim_clip=0.025)
    ax.imshow(img, extent=extent, interpolation='none')

    # plot the glacier outline
    entry_id = fp_raster.parent.name
    gl_sdf = gl_df[gl_df.entry_id == entry_id].to_crs(nc.rio.crs)
    r_gl = gl_sdf.iloc[0]
    gl_sdf.plot(ax=ax, edgecolor=color_list[0], linewidth=line_thickness, facecolor='none')
    inv_date = pd.to_datetime(r_gl.date_inv)
    legend_handles = [
        matplotlib.lines.Line2D(
            [0], [0],
            color=color_list[0],
            label=f'inventory ({inv_date.year})'
        )
    ]

    title = (
        f"{img_date.strftime('%Y-%m-%d')} (Copernicus Sentinel-2, B4-B3-B2)\n"
        f"$A_{{inv}}$ = {r_gl.area_km2:.2f} km$^2$; "
    )

    # get the stats for the current glacier if predictions are available
    if df_results is not None:
        stats = df_results[(df_results.entry_id == entry_id) & (df_results.year == img_date.year)]
        assert len(stats) == 1
        stats = stats.iloc[0]
        recall_debris = stats.recall_debris if (
                (stats.Country == 'CH') & (stats.area_debris / stats.area_inv > 0.01)) else np.nan
        recall_debris_txt = f"{recall_debris * 100:.2f}%" if not np.isnan(recall_debris) else "NA"
        title += (
            f"$A_{{pred}}$ = {stats.area_pred:.2f} ± {stats.area_pred_std:.3f} km$^2$\n"
            f"$FPR_{{20-50m}}$ = {stats['area_non_g_pred_b20_50'] / stats['area_non_g_b20_50'] * 100:.2f}%; "
            f"$recall_{{debris}}$ (CH only) = {recall_debris_txt}"
        )
    ax.set_title(title, fontsize=fontsize)

    # plot the predicted glacier outline if available
    if gl_df_pred is not None:
        gl_sdf_pred = gl_df_pred[gl_df_pred.entry_id == entry_id].to_crs(nc.rio.crs)
        gl_sdf_pred.plot(ax=ax, edgecolor=color_list[1], linewidth=line_thickness, facecolor='none')
        legend_handles.append(
            matplotlib.lines.Line2D(
                [0], [0],
                color=color_list[1],
                label=f'DL4GAM prediction ({img_date.year})'
            )
        )

    # add the legend
    ax.legend(handles=legend_handles, loc='upper left', prop={'size': fontsize - 2})

    # add the scale bar
    ax.add_artist(
        ScaleBar(dx=1.0, length_fraction=0.25, font_properties={'size': fontsize - 2}, location='lower right')
    )

    # Subplot 2) plot the SWIR-NIR-R image
    ax = axes[1]
    img = nc.band_data.isel(band=[band_names.index(b) for b in ('B11', 'B8', 'B4')]).transpose('y', 'x', 'band').values
    img = contrast_stretch(img=img, q_lim_clip=0.025)
    ax.imshow(img, extent=extent, interpolation='none')
    ax.set_title('B11-B8-B4', fontsize=fontsize)

    # Subplot 3) plot the dh/dt (if exists) and the elevation contours
    ax = axes[2]
    img = nc.dhdt.values
    img = contrast_stretch(img=img, q_lim_clip=0.025, scale_to_01=False)
    vmax_abs = max(abs(np.nanmin(img)), abs(np.nanmax(img)))
    p = ax.imshow(img, extent=extent, interpolation='none', cmap='seismic_r', vmin=-vmax_abs, vmax=vmax_abs)
    ax.set_facecolor('gray')  # for the missing data
    cbar = fig.colorbar(p, ax=axes[3], label='dh/dt (m $\\cdot y^{-1}$)', fraction=0.9)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=fontsize - 2)
    range_dhdt = '-'.join([x[-4:] for x in C.EXTRA_RASTERS['dhdt'].parent.name.split('-01-01')[:2]])
    ax.set_title(
        f'dh/dt {range_dhdt} (Hugonnet et al. 2021)\n'
        '(gray = missing)\n'
        ' + COPDEM GLO-30 contours',
        fontsize=fontsize
    )

    # plot the DEM contours
    z = nc.where(nc.mask_crt_g_b20).dem.values
    x, y = np.meshgrid(nc.x.values, nc.y.values)
    z_min = int(np.nanmin(z))
    z_max = int(np.nanmax(z))
    z_step = 100
    z_levels = list(np.arange(int(z_min - z_min % z_step + z_step), z_max - z_max % z_step + 1, z_step))
    z_levels = [z_min] + z_levels if z_levels[0] != z_min else z_levels
    z_levels = z_levels + [z_max] if z_levels[-1] != z_max else z_levels
    cs = ax.contour(x, y, z, levels=z_levels, linewidths=line_thickness, extent=extent)
    ax.clabel(cs, inline=True, fontsize=5, levels=z_levels, use_clabeltext=True)

    # disable the x/y ticks & the border for last plot
    for i, ax in enumerate(axes.flatten()):
        ax.set_xticks([])
        ax.set_yticks([])
        if i > 2:
            ax.axis('off')

    # add a figure title with the glacier ID and the location
    fig.suptitle(
        f"Glacier ID: {r_gl.entry_id.replace('g_', '')} ({r_gl.Country} - {r_gl.LAT:.2f}° N, {r_gl.LON:.2f}° E)",
        fontsize=fontsize + 2,
        x=0.46,
        y=0.98,
    )

    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f'{entry_id}_{img_date.strftime('%Y-%m-%d')}.png')
    plt.close()


if __name__ == "__main__":
    num_cores = 10
    plot_preds = False

    # output root directory
    plot_root_dir = Path('../data/external/scratch_partition/plots/')

    # read the inventory
    print(f"Loading the glacier outlines from {C.GLACIER_OUTLINES_FP}")
    gl_df = gpd.read_file(C.GLACIER_OUTLINES_FP)

    # read the paths to the rasters
    rasters_dir = Path(C.DIR_GL_INVENTORY)
    year = rasters_dir.parent.name
    ds_name = rasters_dir.parent.parent.name
    print(f"Reading the raster paths from {rasters_dir}")
    fp_list = sorted(list(rasters_dir.rglob('*.nc')))
    print(f"rasters_dir = {rasters_dir}; #glaciers = {len(fp_list)}")

    # read the predicted areas and their uncertainties if we want to plot the results
    if plot_preds:
        model_root_dir = Path(f'../data/external/_experiments/{ds_name}/unet/')
        model_version = 'version_0'
        fp_results = model_root_dir / f'df_glacier_agg_{ds_name}_{model_version}.csv'
        print(f"Loading the results from {fp_results}")
        df_results = pd.read_csv(fp_results)

        # read the predicted contours
        fp_pred_outlines = model_root_dir / f'gdf_all_splits/seed_all/{model_version}/{ds_name}/{year}/{year}_pred.shp'
        print(f"Loading the predicted glacier outlines from {fp_pred_outlines}")
        gl_df_pred = gpd.read_file(fp_pred_outlines)

        # keep only the rasters for which we have predictions
        fp_list = [fp for fp in fp_list if fp.parent.name in gl_df_pred.entry_id.values]

        plot_dir = plot_root_dir / ds_name / model_root_dir.name / model_version / year
    else:
        df_results = None
        gl_df_pred = None
        plot_dir = plot_root_dir / ds_name / year

    # plot
    print(f"plot_dir = {plot_dir}")
    _plot_results = partial(
        plot_glacier,
        fig_w_px=1920,
        gl_df=gl_df,
        gl_df_pred=gl_df_pred,
        df_results=df_results,
        plot_dir=plot_dir,
        line_thickness=1,
        fontsize=9,
    )

    run_in_parallel(_plot_results, fp_raster=fp_list, num_procs=num_cores, pbar=True)
