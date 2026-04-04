# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np

from .iou_metric import IoUMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class PlantSegMetric(IoUMetric):
    """Binary metric surface for PlantSeg.

    Outputs:
    - IoU: foreground IoU
    - Dice: foreground Dice
    - Recall: foreground Recall
    - mIoU: mean IoU over background and foreground
    - mACC: mean class accuracy over background and foreground
    """

    def __init__(self,
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = 0,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(
            ignore_index=ignore_index,
            iou_metrics=['mIoU', 'mDice', 'mFscore'],
            nan_to_num=nan_to_num,
            collect_device=collect_device,
            prefix=prefix,
            **kwargs)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        if self.format_only:
            return super().compute_metrics(results)

        results = tuple(zip(*results))
        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta)

        fg_idx = 1
        metrics = OrderedDict()
        metrics['IoU'] = round(float(ret_metrics['IoU'][fg_idx] * 100), 2)
        metrics['Dice'] = round(float(ret_metrics['Dice'][fg_idx] * 100), 2)
        metrics['Recall'] = round(float(ret_metrics['Recall'][fg_idx] * 100), 2)
        metrics['mIoU'] = round(float(np.nanmean(ret_metrics['IoU']) * 100), 2)
        metrics['mACC'] = round(float(np.nanmean(ret_metrics['Acc']) * 100), 2)
        return metrics
