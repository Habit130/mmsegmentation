import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate reproducible train/val splits from trainval.txt')
    parser.add_argument(
        '--data-root',
        default='../IP102',
        help='Path to the IP102 dataset root.')
    parser.add_argument(
        '--trainval-file',
        default='ImageSets/Main/trainval.txt',
        help='Relative path to the source trainval split file.')
    parser.add_argument(
        '--train-file',
        default='ImageSets/Main/train.txt',
        help='Relative path to the generated train split file.')
    parser.add_argument(
        '--val-file',
        default='ImageSets/Main/val.txt',
        help='Relative path to the generated val split file.')
    parser.add_argument(
        '--val-ratio', type=float, default=0.1, help='Validation ratio.')
    parser.add_argument(
        '--seed', type=int, default=3407, help='Random seed for split.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing train.txt and val.txt.')
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    trainval_path = data_root / args.trainval_file
    train_path = data_root / args.train_file
    val_path = data_root / args.val_file

    if not trainval_path.is_file():
        raise FileNotFoundError(f'trainval split file not found: {trainval_path}')
    if train_path.exists() and val_path.exists() and not args.force:
        print(f'Splits already exist: {train_path} and {val_path}')
        return

    names = [
        line.strip() for line in trainval_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(names) < 2:
        raise ValueError('trainval split must contain at least 2 samples.')

    rng = random.Random(args.seed)
    rng.shuffle(names)

    val_count = max(1, int(round(len(names) * args.val_ratio)))
    val_count = min(val_count, len(names) - 1)
    val_names = sorted(names[:val_count])
    train_names = sorted(names[val_count:])

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text('\n'.join(train_names) + '\n', encoding='utf-8')
    val_path.write_text('\n'.join(val_names) + '\n', encoding='utf-8')

    print(f'Generated train split: {train_path} ({len(train_names)} samples)')
    print(f'Generated val split: {val_path} ({len(val_names)} samples)')
    print(f'Seed: {args.seed}, val_ratio: {args.val_ratio}')


if __name__ == '__main__':
    main()
