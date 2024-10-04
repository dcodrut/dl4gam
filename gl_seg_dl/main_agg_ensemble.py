"""
    Aggregate the predictions of the ensemble and export them in a similar directory structure as for the members.
    Additionally, compute the lower and higher bounds of the glacier extent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm

if __name__ == "__main__":
    ds_train = 's2_alps_plus'
    ds_infer = ds_train
    results_root_dir = Path(f'../data/external/experiments_server/{ds_train}/unet')

    seed_list = [42, 43, 44, 45, 46]
    split_list = [1, 2, 3, 4, 5]
    subdirs_infer = ['inv', '2023']
    version = '0'

    for subdir_infer in subdirs_infer:
        for split in split_list:
            # build a dataframe with the paths to the predictions
            df_all = []
            print(f"Preparing the paths to the predictions for split {split} and subdir {subdir_infer}")
            for seed in seed_list:
                model_output_dir = (
                        results_root_dir / f'split_{split}' / f'seed_{seed}' / f'version_{version}' /
                        'output' / 'preds' / ds_infer / subdir_infer / 's_test'
                )
                assert model_output_dir.exists(), f"Model output directory not found: {model_output_dir}"
                fp_list = sorted(list(model_output_dir.glob('**/*.nc')))
                df_fp_crt = pd.DataFrame({
                    'split': split,
                    'seed': seed,
                    'fp': fp_list,
                })
                df_all.append(df_fp_crt)
            df_all = pd.concat(df_all)

            # check if each glacier has the same number of predictions
            df_all['entry_id'] = df_all.fp.apply(lambda x: Path(x).parent.name)
            assert (df_all.groupby(['entry_id']).fp.count() == len(seed_list)).all(), "Predictions missing"
            print(df_all)

            # export the settings (will be needed in the evaluation)
            fp_in = results_root_dir / f'split_{split}' / f'seed_{seed}' / f'version_{version}' / 'settings.yaml'
            # read and change the seed to 'all' (not needed though)
            with open(fp_in, 'r') as fp:
                settings = yaml.safe_load(fp)
                settings['model']['seed'] = 'all'
                fp_out = Path(str(fp_in).replace(f'seed_{seed}', 'seed_all'))
                fp_out.parent.mkdir(exist_ok=True, parents=True)
                with open(fp_out, 'w') as fp:
                    yaml.dump(settings, fp, sort_keys=False)
                    print(f"Settings saved to {fp_out}")

            # compute the average and standard deviation of the predictions over the ensemble
            for entry_id, df_crt in tqdm(df_all.groupby('entry_id'), desc=f"Aggregating predictions"):
                # read the predictions
                nc_list = [xr.open_dataset(fp, decode_coords='all') for fp in df_crt.fp]

                # compute the average and standard deviation; keep the other variables
                nc_avg = nc_list[0].copy()
                for s in ['', '_i_nn', '_i_hypso']:
                    k = f'pred{s}'
                    if k in nc_avg:
                        nc_avg[k] = (('y', 'x'), np.mean([nc[k].values for nc in nc_list], axis=0))
                        nc_avg[k + '_b'] = (nc_avg[k] >= 0.5)
                        nc_avg[k + '_std'] = (('y', 'x'), np.std([nc[k].values for nc in nc_list], axis=0))

                        # compute the lower and higher bounds of the glacier extent
                        nc_avg[f'pred_low_b{s}'] = ((nc_avg[k] - nc_avg[k + '_std']) >= 0.5)
                        nc_avg[f'pred_high_b{s}'] = ((nc_avg[k] + nc_avg[k + '_std']) >= 0.5)

                    # (re)set the CRS
                    for k in nc_avg.data_vars:
                        nc_avg[k].rio.write_crs(nc_avg.rio.crs, inplace=True)

                # export the cube
                model_output_dir = (
                        results_root_dir / f'split_{split}' / 'seed_all' / f'version_{version}' /
                        'output' / 'preds' / ds_infer / subdir_infer / 's_test'
                )
                fp_out = model_output_dir / entry_id / df_crt.fp.iloc[0].name
                fp_out.parent.mkdir(exist_ok=True, parents=True)
                nc_avg.to_netcdf(fp_out)
