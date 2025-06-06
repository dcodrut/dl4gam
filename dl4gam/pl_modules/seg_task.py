import logging
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics as tm
import xarray as xr

# local imports
from pl_modules.data import extract_inputs
from pl_modules.loss import MaskedLoss
from utils.postprocessing import nn_interp, hypso_interp


class GlSegTask(pl.LightningModule):
    def __init__(self, model, task_params, outdir=None, interp='nn'):
        super().__init__()

        self.model = model
        self.loss = MaskedLoss(metric=task_params['loss'])
        self.thr = 0.5
        self.val_metrics = tm.MetricCollection([
            tm.Accuracy(threshold=self.thr, task='binary'),
            tm.JaccardIndex(threshold=self.thr, task='binary'),
            tm.Precision(threshold=self.thr, task='binary'),
            tm.Recall(threshold=self.thr, task='binary'),
            tm.F1Score(threshold=self.thr, task='binary')
        ])
        self.optimizer_settings = task_params['optimization']['optimizer']
        self.lr_scheduler_settings = task_params['optimization']['lr_schedule']

        # get the main logger
        self._logger = logging.getLogger('pytorch_lightning.core')

        self.outdir = outdir
        assert interp in (None, 'nn', 'hypso')
        self.interp = interp

        # initialize the train/val metrics accumulators
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizers = [
            getattr(torch.optim, o['name'])(self.parameters(), **o['args'])
            for o in self.optimizer_settings
        ]
        if self.lr_scheduler_settings is None:
            return optimizers
        schedulers = [
            getattr(torch.optim.lr_scheduler, s['name'])(optimizers[i], **s['args'])
            for i, s in enumerate(self.lr_scheduler_settings)
        ]
        return optimizers, schedulers

    def compute_masked_val_metrics(self, y_pred, y_true, mask):
        # apply the mask for each element in the batch and compute the metrics
        val_metrics_samplewise = []
        for i in range(len(y_true)):
            if mask[i].sum() > 0:
                m = self.val_metrics(preds=y_pred[i][mask[i]], target=y_true[i][mask[i]])
            else:
                m = {k: torch.nan for k in self.val_metrics}
            val_metrics_samplewise.append(m)

        # restructure the output in a single dict {metric: list}
        val_metrics_samplewise = {
            k: torch.tensor([x[k] for x in val_metrics_samplewise], device=y_true.device)
            for k in val_metrics_samplewise[0].keys()
        }

        return val_metrics_samplewise

    def aggregate_step_metrics(self, step_outputs, split_name):
        # extract the glacier name from the patch filepath
        all_fp = [fp for x in step_outputs for fp in x['filepaths']]
        df_per_patch = pd.DataFrame({'fp': all_fp})
        df_per_patch['gid'] = df_per_patch.fp.apply(lambda s: Path(s).parent.name)

        # add the metrics (per patch)
        for m in step_outputs[0]['metrics']:
            col_name = m.replace('Binary', '')
            df_per_patch[col_name] = torch.stack([y for x in step_outputs for y in x['metrics'][m]]).cpu().numpy()

        # prepare the weights based on the glaciers' areas
        _df_weights = pd.DataFrame({
            'gid': df_per_patch.gid,
            'glacier_area': torch.stack([area for x in step_outputs for area in x['glacier_areas']]).cpu().numpy()
        }).groupby('gid').first()
        weights = (_df_weights.glacier_area / _df_weights.glacier_area.sum())

        # average the metrics, both over patches and then over glaciers weighted by their areas
        df_per_glacier = df_per_patch.groupby('gid').mean(numeric_only=True)
        avg_tb_logs = {}
        for m in df_per_glacier.columns:
            avg_tb_logs[f'{m}_{split_name}_epoch_avg'] = df_per_patch[m].mean()
            # skip the weighted version for the debris, doesn't make sense
            if 'debris' not in m:
                avg_tb_logs[f'w_{m}_{split_name}_epoch_avg_per_g'] = (df_per_glacier[m] * weights).sum()

        return avg_tb_logs, df_per_patch

    def training_step(self, batch, batch_idx):
        y_pred = self(batch)
        y_true = batch['mask_all_g'].unsqueeze(dim=1)
        mask = ~batch['mask_no_data'].unsqueeze(dim=1)
        loss = self.loss(preds=y_pred, targets=y_true, mask=mask, samplewise=False)

        tb_logs = {'train_loss': loss}
        self.log_dict(tb_logs, on_epoch=True, on_step=True, batch_size=len(y_true), sync_dist=True)

        # compute the evaluation metrics for each element in the batch
        val_metrics_samplewise = self.compute_masked_val_metrics(y_pred, y_true, mask)

        # compute the recall of the debris-covered areas
        y_true_debris = batch['mask_debris_crt_g'].unsqueeze(dim=1)
        y_true_debris *= y_true  # in case the debris mask contains areas outside the current outlines
        area_debris_fraction = y_true_debris.flatten(start_dim=1).sum(dim=1) / y_true.flatten(start_dim=1).sum(dim=1)
        recall_samplewise_debris = self.compute_masked_val_metrics(y_pred, y_true_debris, mask)['BinaryRecall']
        recall_samplewise_debris[area_debris_fraction < 0.01] = torch.nan
        val_metrics_samplewise['BinaryRecall_debris'] = recall_samplewise_debris

        res = {
            'loss': loss,
            'metrics': val_metrics_samplewise,
            'filepaths': batch['fp'],
            'glacier_areas': batch['glacier_area']
        }

        self.training_step_outputs.append(res)

        return res

    def on_train_epoch_start(self):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2189
        print('\n')

    def on_train_epoch_end(self):
        avg_tb_logs, df = self.aggregate_step_metrics(self.training_step_outputs, split_name='train')

        # show the epoch as the x-coordinate
        avg_tb_logs['step'] = float(self.current_epoch)
        self.log_dict(avg_tb_logs, on_step=False, on_epoch=True, sync_dist=True)

        # clear the accumulator
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        y_pred = self(batch)
        y_true = batch['mask_all_g'].unsqueeze(dim=1)
        mask = ~batch['mask_no_data'].unsqueeze(dim=1)
        loss_samplewise = self.loss(preds=y_pred, targets=y_true, mask=mask, samplewise=True)

        # compute the evaluation metrics for each element in the batch
        val_metrics_samplewise = self.compute_masked_val_metrics(y_pred, y_true, mask)

        # compute the recall of the debris-covered areas (if the area is above 1%)
        y_true_debris = batch['mask_debris_crt_g'].unsqueeze(dim=1)
        y_true_debris *= y_true  # in case the debris mask contains areas outside the current outlines
        area_debris_fraction = y_true_debris.flatten(start_dim=1).sum(dim=1) / y_true.flatten(start_dim=1).sum(dim=1)
        recall_samplewise_debris = self.compute_masked_val_metrics(y_pred, y_true_debris, mask)['BinaryRecall']
        recall_samplewise_debris[area_debris_fraction < 0.01] = torch.nan
        val_metrics_samplewise['BinaryRecall_debris'] = recall_samplewise_debris

        # add also the loss to the metrics
        val_metrics_samplewise.update({'loss': loss_samplewise})

        tb_logs = {'val_loss': loss_samplewise.mean()}
        self.log_dict(tb_logs, on_epoch=True, on_step=True, batch_size=len(y_true), sync_dist=True)

        res = {'metrics': val_metrics_samplewise, 'filepaths': batch['fp'], 'glacier_areas': batch['glacier_area']}
        self.validation_step_outputs.append(res)

        return res

    def on_validation_epoch_end(self):
        avg_tb_logs, df = self.aggregate_step_metrics(self.validation_step_outputs, split_name='val')

        # show the stats
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', None):
            self._logger.info(f'validation scores stats:\n{df.describe().round(3)}')
            self._logger.info(f'validation scores stats (per glacier):\n'
                              f'{df.groupby("gid").mean(numeric_only=True).describe().round(3)}')

        # export the stats if needed
        if self.outdir is not None:
            self.outdir = Path(self.outdir)
            self.outdir.mkdir(parents=True, exist_ok=True)
            fp = self.outdir / 'stats.csv'
            df.to_csv(fp)
            self._logger.info(f'Stats exported to {str(fp)}')
            fp = self.outdir / 'stats_avg_per_glacier.csv'
            df.groupby('gid').mean(numeric_only=True).to_csv(fp)
            self._logger.info(f'Stats per glacier exported to {str(fp)}')

        # show the epoch as the x-coordinate
        avg_tb_logs['step'] = float(self.current_epoch)
        self.log_dict(avg_tb_logs, on_step=False, on_epoch=True, sync_dist=True)

        # clear the accumulator
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        y_pred = self(batch)
        y_true = batch['mask_all_g'].unsqueeze(dim=1)
        mask = ~batch['mask_no_data'].unsqueeze(dim=1)
        loss_samplewise = self.loss(preds=y_pred, targets=y_true, mask=mask, samplewise=True)

        # compute the evaluation metrics for each element in the batch
        val_metrics_samplewise = self.compute_masked_val_metrics(y_pred, y_true, mask)

        # add also the loss to the metrics
        val_metrics_samplewise.update({'loss': loss_samplewise})

        res = {'metrics': val_metrics_samplewise, 'filepaths': batch['fp'], 'glacier_areas': batch['glacier_area']}
        res['patch_info'] = batch['patch_info']
        res['preds'] = y_pred

        self.test_step_outputs.append(res)

        return res

    def on_test_epoch_end(self):
        # collect all filepaths
        filepaths = [y for x in self.test_step_outputs for y in x['filepaths']]

        # ensure that all the predictions are for the same glacier
        if len(set(filepaths)) > 1:
            raise NotImplementedError
        cube_fp = filepaths[0]

        # read the original glacier nc and create accumulators based on its shape
        nc = xr.load_dataset(cube_fp, decode_coords='all')
        preds_acc = torch.zeros(nc.mask_crt_g.shape, device=self.device)
        preds_cnt = torch.zeros(size=preds_acc.shape, device=self.device)
        prc_margin_to_drop = 0.05  # percentage of the patch size to drop from each side (to avoid border effects)
        for j in range(len(self.test_step_outputs)):
            preds = self.test_step_outputs[j]['preds']
            patch_infos = self.test_step_outputs[j]['patch_info']

            batch_size = preds.shape[0]
            for i in range(batch_size):
                crt_pred = preds[i][0]

                # get the (pixel) coordinates of the current patch
                minx, maxx = patch_infos['minx'][i], patch_infos['maxx'][i]
                miny, maxy = patch_infos['miny'][i], patch_infos['maxy'][i]

                # drop the margins
                dx = int(crt_pred.shape[1] * prc_margin_to_drop)
                minx, maxx = minx + dx, maxx - dx
                miny, maxy = miny + dx, maxy - dx  # square patches
                crt_pred = crt_pred[dx:-dx, dx:-dx]

                preds_acc[miny:maxy, minx:maxx] += crt_pred
                preds_cnt[miny:maxy, minx:maxx] += 1

        preds_acc /= preds_cnt

        # copy to CPU memory
        preds_acc_np = preds_acc.cpu().numpy()

        # store the predictions as xarray based on the original nc
        nc_pred = nc.copy()
        nc_pred['pred'] = (('y', 'x'), preds_acc_np)
        nc_pred['pred_b'] = (('y', 'x'), preds_acc_np >= self.thr)

        # fill-in the masked pixels (only within 50m buffer) using two different methods
        mask_to_fill = extract_inputs(ds=nc, fp=cube_fp, input_settings=self.model.input_settings)['mask_no_data']
        data = nc_pred.pred.values
        mask_crt_g_b50 = (nc.mask_crt_g_b50.values == 1)
        mask_to_fill &= mask_crt_g_b50
        mask_ok = (~mask_to_fill) & mask_crt_g_b50

        if self.interp is not None:
            n_px = 30  # how many pixels to use as source for interpolation value
            if mask_to_fill.sum() > 0 and mask_ok.sum() >= n_px:
                # make a copy of the original probs in case we want to analyze the impact of the interpolation later
                nc_pred['pred_orig'] = (('y', 'x'), data)

                # use the nearest neighbours
                if self.interp == 'nn':
                    data_interp = nn_interp(data=data, mask_to_fill=mask_to_fill, mask_ok=mask_ok, num_nn=n_px)
                elif self.interp == 'hypso':
                    # use the pixels with the closest elevations
                    dem = nc_pred.dem.values
                    data_interp = hypso_interp(
                        data=data, mask_to_fill=mask_to_fill, mask_ok=mask_ok, dem=dem, num_px=n_px
                    )
                nc_pred['pred'] = (('y', 'x'), data_interp)
                nc_pred['pred_b'] = (('y', 'x'), data_interp >= self.thr)

        # add the CRS to the new data arrays
        for c in [_c for _c in nc_pred.data_vars if 'pred' in _c]:
            nc_pred[c].rio.write_crs(nc_pred.rio.crs, inplace=True)

        # drop the data variables to save space
        nc_pred = nc_pred[[c for c in nc_pred.data_vars if 'pred' in c or 'mask' in c]]

        gl_id = Path(cube_fp).parent.name
        cube_pred_fp = Path(self.outdir) / gl_id / Path(cube_fp).name
        cube_pred_fp.parent.mkdir(parents=True, exist_ok=True)
        cube_pred_fp.unlink(missing_ok=True)
        nc_pred.to_netcdf(cube_pred_fp)
        self._logger.info(f'Cube with predictions exported to {cube_pred_fp}')

        # clear the accumulator
        self.test_step_outputs.clear()
