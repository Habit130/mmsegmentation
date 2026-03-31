# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class IP102BinaryDataset(BaseSegDataset):
    """IP102 binary segmentation dataset.

    This dataset reads images from ``JPEGImages`` and masks from ``Masks``.
    The raw masks use pixel values ``0`` for background and ``255`` for
    foreground, so the annotation loader must remap ``255 -> 1``.
    """

    METAINFO = dict(
        classes=('background', 'foreground'),
        palette=[[0, 0, 0], [255, 255, 255]])

    LABEL_MAP = {0: 0, 255: 1}

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        self._metainfo['label_map'] = copy.deepcopy(self.LABEL_MAP)
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args)

    def load_data_list(self):
        data_list = super().load_data_list()
        for data_info in data_list:
            data_info['label_map'] = copy.deepcopy(self.LABEL_MAP)
        return data_list
