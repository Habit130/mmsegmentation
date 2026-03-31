import argparse
from pathlib import Path

from mmengine.config import Config

from mmseg.registry import DATASETS, MODELS


DEFAULT_CONFIGS = [
    'configs/ip102/upernet_swin-t_8xb2-160k_ip102-512x512.py',
    'configs/ip102/ocrnet_hr18_4xb4-160k_ip102-512x512.py',
    'configs/ip102/hrnet_w18_4xb4-160k_ip102-512x512.py',
    'configs/ip102/segmenter_vit-b16_mask_8xb1-160k_ip102-512x512.py',
    'configs/ip102/fcn_r50-d8_4xb4-160k_ip102-512x512.py',
    'configs/ip102/pspnet_r50-d8_4xb4-160k_ip102-512x512.py',
    'configs/ip102/deeplabv3plus_r50-d8_4xb4-160k_ip102-512x512.py',
    'configs/ip102/segformer_mit-b2_8xb2-160k_ip102-512x512.py',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate that IP102 configs can build models and datasets.')
    parser.add_argument(
        'configs',
        nargs='*',
        default=DEFAULT_CONFIGS,
        help='Config files to validate.')
    return parser.parse_args()


def main():
    args = parse_args()
    for config_path in args.configs:
        cfg = Config.fromfile(config_path)
        MODELS.build(cfg.model)
        DATASETS.build(cfg.train_dataloader.dataset)
        DATASETS.build(cfg.val_dataloader.dataset)
        DATASETS.build(cfg.test_dataloader.dataset)
        print(f'OK: {Path(config_path).resolve()}')


if __name__ == '__main__':
    main()
