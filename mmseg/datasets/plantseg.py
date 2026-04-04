# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PlantSegDataset(BaseSegDataset):
    """PlantSeg dataset backed by a sibling ``main.json`` manifest.

    The original dataset stores binary masks with values ``0`` and ``255``.
    Label remapping is handled in the dedicated annotation loader so that the
    raw files remain untouched.
    """

    METAINFO = dict(
        classes=('background', 'foreground'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 split='train',
                 main_json='main.json',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        self.split = split
        self.main_json = main_json
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load sample records from ``main.json``."""
        main_json = self.main_json
        if self.data_root is not None and not osp.isabs(main_json):
            main_json = osp.join(self.data_root, main_json)

        with open(main_json, encoding='utf-8') as f:
            records = json.load(f)

        data_list = []
        for record in records:
            if record.get('split') != self.split:
                continue

            img_path = record['image']
            seg_map_path = record.get('mask')
            if self.data_root is not None:
                img_path = osp.join(self.data_root, img_path)
                if seg_map_path is not None:
                    seg_map_path = osp.join(self.data_root, seg_map_path)

            data_info = dict(
                img_path=img_path,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[])

            if seg_map_path is not None:
                data_info['seg_map_path'] = seg_map_path

            if 'id' in record:
                data_info['sample_id'] = record['id']
            if 'disease_label' in record:
                data_info['disease_label'] = record['disease_label']
            if 'caption' in record:
                data_info['caption'] = record['caption']

            data_list.append(data_info)

        return data_list
