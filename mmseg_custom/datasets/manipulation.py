import warnings
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv import print_log
from mmseg.datasets.builder import DATASETS
from .custom import FixedCustomDataset
from prettytable import PrettyTable
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve

from ..core.evaluation.metrics import pre_eval_to_metrics, intersect_and_union, eval_metrics
from .pipelines.formatting import ToBinaryMask


class ManipulationDetectionDataset(FixedCustomDataset):
    CLASSES = ('Real', 'Fake')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, img_suffix, seg_map_suffix, **kwargs):
        super(ManipulationDetectionDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)

        self.f1_score = f1_score
        self.roc_auc_score = roc_auc_score

        self.binarize = ToBinaryMask()

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        self.binarize(results)
        return results['gt_semantic_seg']

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            self.binarize(results)
            yield results['gt_semantic_seg']

    def pre_eval_with_logit(self, preds, logits, indices):
        """Collect eval result from each iteration.

        Args:
            logit:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]
        if not isinstance(logits, list):
            logits = [logits]

        pre_eval_results = []
        for pred, logit, index in zip(preds, logits, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            seg_map_flatten = seg_map.flatten()
            logit_flatten = logit.flatten()

            def calculate_eer_f1(seg_map_flatten, logit_flatten):
                fpr, tpr, threshold = roc_curve(seg_map_flatten, logit_flatten, pos_label=1)
                fnr = 1 - tpr
                eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
                pred_eer = np.where(logit_flatten > eer_threshold, 1, 0)
                f1_eer = self.f1_score(seg_map_flatten, pred_eer)
                return f1_eer

            def calculate_best_f1(seg_map_flatten, logit_flatten):
                precision, recall, thresholds = precision_recall_curve(seg_map_flatten, logit_flatten, pos_label=1)
                f1_scores = 2 * recall * precision / (recall + precision + 1e-6)
                return np.max(f1_scores)

            f1_eer = calculate_eer_f1(seg_map_flatten, logit_flatten)
            f1_fix = self.f1_score(seg_map_flatten, pred.flatten())
            f1_best = calculate_best_f1(seg_map_flatten, logit_flatten)

            auc = self.roc_auc_score(seg_map_flatten, logit_flatten)
            auc_pscc = max(auc, 1 - auc)

            class_metrics = intersect_and_union(
                pred,
                seg_map,
                len(self.CLASSES),
                self.ignore_index,
                # as the labels has been converted when dataset initialized
                # in `get_palette_for_custom_classes ` this `label_map`
                # should be `dict()`, see
                # https://github.com/open-mmlab/mmsegmentation/issues/1415
                # for more ditails
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label)
            pre_eval_results.append((*class_metrics, f1_eer, f1_fix, f1_best, auc, auc_pscc))

        return pre_eval_results

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics: dict = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics = dict((key, val) for key, val in ret_metrics.items() if val.size > 1)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results


@DATASETS.register_module()
class CASIADataset(ManipulationDetectionDataset):
    def __init__(self, **kwargs):
        super(CASIADataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_gt.png',
            **kwargs)


@DATASETS.register_module()
class IMD20Dataset(ManipulationDetectionDataset):
    def __init__(self, **kwargs):
        super(IMD20Dataset, self).__init__(
            img_suffix=('.jpg', '.png'),
            seg_map_suffix='_mask.png',
            **kwargs)


@DATASETS.register_module()
class ColumbiaDataset(ManipulationDetectionDataset):
    def __init__(self, **kwargs):
        super(ColumbiaDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='_edgemask.png',
            **kwargs)


@DATASETS.register_module()
class PSCCMixed(ManipulationDetectionDataset):
    CLASSES = ('Real', 'Fake')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(PSCCMixed, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)


@DATASETS.register_module()
class NIST16Dataset(ManipulationDetectionDataset):
    def __init__(self, **kwargs):
        super(NIST16Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.jpg',
            **kwargs)

@DATASETS.register_module()
class CoverageDataset(ManipulationDetectionDataset):
    def __init__(self, **kwargs):
        super(CoverageDataset, self).__init__(
            img_suffix='t.tif',
            seg_map_suffix='forged.tif',
            **kwargs)
