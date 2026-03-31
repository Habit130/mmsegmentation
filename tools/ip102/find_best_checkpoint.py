import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find the best checkpoint in a work directory.')
    parser.add_argument('--work-dir', required=True, help='Model work directory.')
    parser.add_argument(
        '--metric', default='mIoU', help='Metric suffix used in best checkpoint names.')
    return parser.parse_args()


def main():
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    candidates = sorted(work_dir.glob(f'best_{args.metric}_*.pth'))
    if not candidates:
        candidates = sorted(work_dir.glob('best_*.pth'))
    if candidates:
        print(str(candidates[-1]))
        return

    last_checkpoint_file = work_dir / 'last_checkpoint'
    if last_checkpoint_file.is_file():
        ckpt = last_checkpoint_file.read_text(encoding='utf-8').strip()
        if ckpt:
            ckpt_path = Path(ckpt)
            if not ckpt_path.is_absolute():
                ckpt_path = (work_dir / ckpt_path).resolve()
            print(str(ckpt_path))
            return

    latest = work_dir / 'latest.pth'
    if latest.is_file():
        print(str(latest))
        return

    raise FileNotFoundError(f'No checkpoint found under {work_dir}')


if __name__ == '__main__':
    main()
