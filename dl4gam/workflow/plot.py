import logging
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cmap
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
from dl4gam.configs.config import BaseDatasetCfg
from dl4gam.utils import run_in_parallel, contrast_stretch, extract_date_from_fn

log = logging.getLogger(__name__)


def plot_glacier(
        fp_raster: Path,
        plot_dir: Path,
        gdf_gt: gpd.GeoDataFrame,
        gdf_pred: Optional[gpd.GeoDataFrame] = None,
        df_pred_stats: Optional[pd.DataFrame] = None,
        bands_img_1: Tuple[str, str, str] = ('R', 'G', 'B'),
        bands_img_2: Tuple[str, str, str] = ('SWIR', 'NIR', 'R'),
        plot_dhdt: bool = True,
        dhdt_time_range: Optional[str] = None,
        source_name: str = 'not specified',
        fig_w_px: int = 1920,
        dpi: int = 150,
        line_thickness: float = 1.0,
        fontsize: int = 12,
        q_lim_contrast: float = 0.02,  # default in QGIS
):
    ds = xr.open_dataset(fp_raster, decode_coords='all')
    img_date = pd.to_datetime(extract_date_from_fn(fp_raster.stem))

    # Set the width ratios (the 3 images + the colorbar) and prepare the figure
    width_ratios = [1, 1, 1, 0.035]

    # Resample s.t. the image width fits the desired figure width
    img_w_px = int((fig_w_px - width_ratios[-1] * fig_w_px) / 3)
    scale_factor = img_w_px / len(ds.x)
    img_h_px = int(len(ds.y) * scale_factor)
    ds = ds.rio.reproject(dst_crs=ds.rio.crs, shape=(img_h_px, img_w_px), resampling=rasterio.enums.Resampling.bilinear)
    extent = [float(x) for x in [ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()]]

    # Prepare the figure
    fig_w = fig_w_px / dpi
    h_title_px = img_w_px * 0.2
    fig_h = (img_h_px + h_title_px) / dpi
    fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h), dpi=dpi, width_ratios=width_ratios, layout='constrained')

    # Subplot 1) plot the RGB image with the glacier outlines
    ax = axes[0]
    band_names = ds.band_data.long_name
    img = ds.band_data.isel(band=[band_names.index(b) for b in bands_img_1]).transpose('y', 'x', 'band').values
    img = contrast_stretch(img=img, q_lim_clip=q_lim_contrast)
    ax.imshow(img, extent=extent, interpolation='none')

    # Plot the glacier outline
    entry_id = fp_raster.parent.name
    sgdf_gt = gdf_gt[gdf_gt.entry_id == entry_id].to_crs(ds.rio.crs)
    color_gt = cmap.Color('gold').hex
    sgdf_gt.plot(ax=ax, edgecolor=color_gt, linewidth=line_thickness, facecolor='none')
    r_gl = sgdf_gt.iloc[0]
    inv_date = pd.to_datetime(r_gl.date_inv)
    legend_handles = [
        matplotlib.lines.Line2D(
            [0], [0],
            color=color_gt,
            label=f'inventory ({inv_date.year})'
        )
    ]

    # Get the stats for the current glacier if predictions are available
    if df_pred_stats is not None:
        stats = df_pred_stats.loc[entry_id]
        # Compute the debris-specific recall only for Switzerland and if debris coverage is >= 1% (to avoid noise)
        recall_debris = stats.TPR_debris if (
                ('TPR_debris' in stats)
                & (r_gl.country_code == 'CH')
                & (stats.debris_coverage_f >= 0.01)
        ) else np.nan
        recall_debris_txt = f"{recall_debris * 100:.2f}%" if not np.isnan(recall_debris) else "NA"
        rel_err = (stats['FP'] + stats['FN']) / stats['P']
        pred_err_txt = f"± {stats.area_pred_std:.3f}" if 'area_pred_std' in stats else ""
        title = (
            f"$A_{{pred}}$ = {stats.pred:.2f}{pred_err_txt} km$^2$\n"
            f"$A_{{P}}$ = {stats['P']:.2f} km$^2$, $A_{{TP}}$ = {stats['TP']:.2f} km$^2$ (TPR = {stats['TPR']:.1%})\n"
            f"$A_{{N}}$ = {stats['N']:.2f} km$^2$, $A_{{FP}}$ = {stats['FP']:.2f} km$^2$ (FPR = {stats['FPR']:.1%})\n"
            f"rel. err.: $(FP + FN) / P$ = {rel_err:.2%}; "
            f"$TPR_{{debris}}$ (CH only) = {recall_debris_txt}"
        )
    else:
        title = f"$A_{{inv}}$ = {r_gl.area_km2:.2f} km$^2$"

    ax.set_title(title, fontsize=fontsize)

    # Plot the predicted glacier outline if available
    if gdf_pred is not None:
        sgdf_pred = gdf_pred[gdf_pred.index == entry_id].to_crs(ds.rio.crs)
        color_pred = cmap.Color('deeppink').hex
        if not sgdf_pred.geometry.iloc[0].is_empty:
            sgdf_pred.plot(ax=ax, edgecolor=color_pred, linewidth=line_thickness, facecolor='none')

        # Still add the legend entry even if the geometry is empty (i.e. no prediction)
        legend_handles.append(
            matplotlib.lines.Line2D(
                [0], [0],
                color=color_pred,
                label=f'DL4GAM prediction ({img_date.year})'
            )
        )

    # Add the legend
    ax.legend(handles=legend_handles, loc='upper left', prop={'size': fontsize - 2})

    # Add the scale bar
    ax.add_artist(
        ScaleBar(dx=1.0, length_fraction=0.25, font_properties={'size': fontsize - 2}, location='lower right')
    )

    # Subplot 2) plot the SWIR-NIR-R image
    ax = axes[1]
    img = ds.band_data.isel(band=[band_names.index(b) for b in bands_img_2]).transpose('y', 'x', 'band').values
    img = contrast_stretch(img=img, q_lim_clip=q_lim_contrast)
    ax.imshow(img, extent=extent, interpolation='none')
    ax.set_title(
        f"left: {'-'.join(bands_img_1)}, below: {'-'.join(bands_img_2)}\n"
        f"Source: {source_name}",
        fontsize=fontsize
    )

    # Subplot 3) plot the dh/dt (if exists, the DEM if not) and the elevation contours
    ax = axes[2]
    cmap_contours = cmap.Colormap('crameri:batlow').to_mpl()
    if plot_dhdt:
        if 'dhdt' not in ds.data_vars:
            raise ValueError(f"dhdt not found in {fp_raster}. Please check the dataset configuration.")
        img = ds.dhdt.values
        img = contrast_stretch(img=img, q_lim_clip=q_lim_contrast, scale_to_01=False)
        vmax_abs = max(abs(np.nanmin(img)), abs(np.nanmax(img)))
        _cmap = cmap.Colormap('vik_r').to_mpl()
        p = ax.imshow(img, extent=extent, interpolation='none', cmap=_cmap, vmin=-vmax_abs, vmax=vmax_abs)
        ax.set_facecolor('gray')  # for the missing data
    title = ''
    if plot_dhdt:
        cbar = fig.colorbar(p, ax=axes[3], label='dh/dt (m $\\cdot y^{-1}$)', fraction=0.9)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        title = (
            f"dh/dt {dhdt_time_range} (Hugonnet et al. 2021)"
            f"\n(gray = missing)\n+ glacier-wide COPDEM GLO-30 contours")
    else:
        img = ds.dem.values
        p = ax.imshow(img, extent=extent, interpolation='none', cmap=cmap_contours)
        cbar = fig.colorbar(p, ax=axes[3], label='elevation (m a.s.l.)', fraction=0.9)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        title = f'Glacier-wide elevation contours (COPDEM GLO-30)'

    cbar.ax.tick_params(labelsize=fontsize - 2)
    ax.set_title(title, fontsize=fontsize)

    # plot the DEM contours in any case
    z = ds.where(ds.mask_infer).dem.values
    x, y = np.meshgrid(ds.x.values, ds.y.values)
    z_min = int(np.nanmin(z))
    z_max = int(np.nanmax(z))
    z_step = 100
    z_levels = list(np.arange(int(z_min - z_min % z_step + z_step), z_max - z_max % z_step + 1, z_step))
    z_levels = [z_min] + z_levels if z_levels[0] != z_min else z_levels
    z_levels = z_levels + [z_max] if z_levels[-1] != z_max else z_levels
    _c = 'black' if not plot_dhdt else None
    _cmap = None if not plot_dhdt else 'terrain'
    cs = ax.contour(x, y, z, levels=z_levels, linewidths=line_thickness, extent=extent, cmap=_cmap, colors=_c)
    texts = ax.clabel(cs, inline=True, fontsize=6, levels=z_levels, use_clabeltext=True)
    for txt in texts:
        txt.set_weight('bold')

    # disable the x/y ticks & the border for last plot
    for i, ax in enumerate(axes.flatten()):
        ax.set_xticks([])
        ax.set_yticks([])
        if i > 2:
            ax.axis('off')

    # add a figure title with the glacier ID and the location
    fig.suptitle(
        f"Glacier ID: {r_gl.entry_id} ({r_gl.country_code} - {r_gl.cenlat:.2f}° N, {r_gl.cenlon:.2f}° E)\n"
        f"Date: {img_date.strftime('%Y-%m-%d')}\n",
        fontsize=fontsize + 2,
        x=0.48
    )

    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f'{entry_id}_{img_date.strftime('%Y-%m-%d')}.png')
    plt.close()


def main(
        dataset_cfg: BaseDatasetCfg,
        fold: Optional[str] = None,
        checkpoint_dir: Optional[str | Path] = None,
):
    """
    Plot the images and, optionally, the predicted glacier outlines and statistics.

    We plot the RGB image in the first panel, together with the glacier outlines from the inventory and, if available,
    the predicted outlines. The second panel shows, depending on the dataset, either the SWIR-NIR-R image or the
    NIR-R-G image. If none of the bands are available, we will again show the RGB image. This panel has no annotations
    to make it easier to check the visual cues. The third panel shows the dh/dt (if available) and the elevation
    contours from the DEM.

    :param dataset_cfg: Configuration object for the dataset, which contains paths to the glacier outlines and cubes.
    :param checkpoint_dir: Directory where the model predictions are stored. If None, only the images will be plotted.
    :param fold: Fold name for the predictions (e.g., 'train', 'valid', 'test').
    :return: None
    """

    ds = dataset_cfg

    # Read the glacier inventory
    log.info(f"Reading the glacier outlines from {ds.geoms_fp}")
    gdf_gt = gpd.read_file(ds.geoms_fp, layer='glacier_sel')

    # Prepare the paths to the glacier cubes
    fp_list = sorted(list(Path(ds.cubes_dir).rglob('*.nc')))
    log.info(f"Found {len(fp_list)} glacier cubes in {ds.cubes_dir}")

    # Read the predicted outlines (including the TP, FP, FN geometries)
    if checkpoint_dir is not None:
        ds_subset = f"dataset={ds.name}_year={ds.year}_fold={fold}"
        fp_geoms_eval = checkpoint_dir.parent / 'eval' / f"geoms_{ds_subset}.gpkg"
        log.info(f"Loading the predicted geometries (unbounded) from {fp_geoms_eval} (layer 'pred')")
        gdf_pred = gpd.read_file(fp_geoms_eval, layer='pred').set_index('entry_id')

        fp_stats = checkpoint_dir.parent / 'eval' / f"stats_{ds_subset}.csv"
        log.info(f"Loading the prediction statistics from {fp_stats}")
        df_stats = pd.read_csv(fp_stats, index_col='entry_id')

        # Keep only the rasters for which we have predictions
        fp_list = [fp for fp in fp_list if fp.parent.name in gdf_pred.index.values]
        log.info(f"Keeping {len(fp_list)} glacier cubes for which we have predictions")
        plot_dir = checkpoint_dir.parent / 'plots_with_preds' / ds_subset
    else:
        df_stats = None
        gdf_pred = None
        plot_dir = Path(ds.plots_dir)

    log.info(f"Plotting to {plot_dir}")

    # Set up which bands to use for the plots
    _bands = list(ds.raw_data.bands_rename.values()) if ds.raw_data.bands_rename is not None else ds.raw_data.bands
    bands_img_1 = ('R', 'G', 'B')
    if {'SWIR', 'NIR', 'R'}.issubset(_bands):
        bands_img_2 = ('SWIR', 'NIR', 'R')
    elif {'NIR', 'R', 'G'}.issubset(_bands):
        bands_img_2 = ('NIR', 'R', 'G')
    else:
        bands_img_2 = bands_img_1
    log.info(f"Using bands {bands_img_1} for the first panel and {bands_img_2} for the second one")

    # Get the range of the dhdt if available
    plot_dhdt = 'dhdt' in ds.extra_rasters
    if plot_dhdt:
        dhdt_time_range = '-'.join([x[-4:] for x in Path(ds.extra_rasters['dhdt']).parent.name.split('-01-01')[:2]])
    else:
        dhdt_time_range = None

    # Plot in parallel
    _plot_results = partial(
        plot_glacier,
        fig_w_px=2560,
        gdf_gt=gdf_gt,
        gdf_pred=gdf_pred,
        df_pred_stats=df_stats,
        plot_dir=plot_dir,
        bands_img_1=bands_img_1,
        bands_img_2=bands_img_2,
        source_name=ds.source_name,
        plot_dhdt='dhdt' in ds.extra_rasters,
        dhdt_time_range=dhdt_time_range,
        line_thickness=1,
        fontsize=9,
        q_lim_contrast=0.035
    )
    run_in_parallel(_plot_results, fp_raster=fp_list)
