import argparse
import json
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run test and dump structured IP102 metrics.')
    parser.add_argument('config', help='Config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file path.')
    parser.add_argument('--work-dir', help='Work directory.')
    parser.add_argument(
        '--metrics-file',
        help='Output JSON path for structured metrics. Defaults to <work_dir>/final_test_metrics.json')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config settings.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher.')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    metrics_file = args.metrics_file or osp.join(cfg.work_dir,
                                                 'final_test_metrics.json')
    cfg.test_evaluator['output_metrics_path'] = metrics_file

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    with open(metrics_file, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    payload['checkpoint'] = osp.abspath(args.checkpoint)
    payload['config'] = osp.abspath(args.config)
    payload['work_dir'] = osp.abspath(cfg.work_dir)
    payload['summary'].update({key: float(val) for key, val in metrics.items()})
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(metrics_file)


if __name__ == '__main__':
    main()
