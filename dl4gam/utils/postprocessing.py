import numpy as np


def nn_interp(data, mask_to_fill, mask_ok, num_nn=100):
    y_px_to_fill, x_px_to_fill = np.where(mask_to_fill)
    y_px_ok, x_px_ok = np.where(mask_ok)

    data_interp = np.zeros_like(data) + np.nan
    data_interp[mask_ok] = data[mask_ok]
    n_ok = len(x_px_ok)
    for x, y in zip(x_px_to_fill, y_px_to_fill):
        # get the closest num_nn pixels
        dists = (x - x_px_ok) ** 2 + (y - y_px_ok) ** 2

        # keep only the closest pixels
        max_dist = np.quantile(dists, q=num_nn/n_ok)
        idx = (dists <= max_dist)
        x_px_ok_sel, y_px_ok_sel = x_px_ok[idx], y_px_ok[idx]

        # compute the mean over the selected pixels
        fill_value = np.mean(data[y_px_ok_sel, x_px_ok_sel])
        data_interp[y, x] = fill_value
    return data_interp


def hypso_interp(data, mask_to_fill, mask_ok, dem, num_px=100):
    # get the unique elevation values that have to be filled in
    h_to_fill_sorted = np.sort(np.unique(dem[mask_to_fill]))

    # prepare a sorted array of filled elevations which will be used to get the interpolation
    h_ok_sorted = np.sort(dem[mask_ok])

    data_interp = np.zeros_like(data) + np.nan
    data_interp[mask_ok] = data[mask_ok]
    for h in h_to_fill_sorted:
        # get the closest ~num_px based on their elevation
        i_h = np.searchsorted(h_ok_sorted, h)
        i_h_min = max(0, i_h - num_px // 2)
        i_h_max = min(len(h_ok_sorted) - 1, i_h + num_px // 2)
        h_min = h_ok_sorted[i_h_min]
        h_max = h_ok_sorted[i_h_max]

        crt_mask_h_ok = mask_ok & (h_min <= dem) & (dem <= h_max)
        crt_mask_h_to_fill = mask_to_fill & (dem == h)
        fill_value = np.mean(data[crt_mask_h_ok])
        data_interp[crt_mask_h_to_fill] = fill_value

    return data_interp
