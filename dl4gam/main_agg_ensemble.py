"""
    Aggregate the predictions of the ensemble and export them in a similar directory structure as for the members.
    Additionally, compute the lower and higher bounds of the glacier extent.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_root_dir', type=str, metavar='path/to/model_root_dir',
                        help='root directory of the model', required=True)
    parser.add_argument('--split', type=str, metavar='split_$SPLIT', help='split', required=True)
    parser.add_argument('--version', type=str, metavar='version_$VERSION/', help='version', required=True)
    parser.add_argument('--dataset_name', type=str, metavar='dataset_name', help='dataset name', required=True)
    parser.add_argument('--eval_subdir', type=str, metavar='eval_subdir', help='evaluation subdir', required=True)
    parser.add_argument('--fold', type=str, metavar='s_test', help='fold', required=True)
    parser.add_argument('--seed_list', nargs='+', metavar='seed_list', help='seed list', required=True)
    args = parser.parse_args()

    results_root_dir = Path(args.model_root_dir)
    ds_infer = args.dataset_name
    split = args.split
    version = args.version
    subdir_infer = args.eval_subdir
    fold = args.fold
    seed_list = list(args.seed_list)

    # build a dataframe with the paths to the predictions
    df_all = []
    for seed in seed_list:
        model_output_dir = (
                results_root_dir / split / f'seed_{seed}' / version / 'output' / 'preds' /
                ds_infer / subdir_infer / fold
        )
        print(f"Loading predictions paths from {model_output_dir}")
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

    # export the settings of one of the models (will be needed in the evaluation)
    # read the settings and change the seed to 'all' (not needed though)
    fp_in = results_root_dir / split / f'seed_{seed}' / version / 'settings.yaml'
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
                results_root_dir / split / 'seed_all' / version / 'output' / 'preds' / ds_infer / subdir_infer / fold
        )
        fp_out = model_output_dir / entry_id / df_crt.fp.iloc[0].name
        fp_out.parent.mkdir(exist_ok=True, parents=True)
        nc_avg.to_netcdf(fp_out)
