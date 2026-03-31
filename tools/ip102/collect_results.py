import argparse
import csv
import json
from pathlib import Path


MODEL_SPECS = [
    ('upernet_swin-t_8xb2-160k_ip102-512x512', 'UPerNet Swin-T'),
    ('ocrnet_hr18_4xb4-160k_ip102-512x512', 'OCRNet HRNet-W18'),
    ('hrnet_w18_4xb4-160k_ip102-512x512', 'HRNet HRNet-W18'),
    ('segmenter_vit-b16_mask_8xb1-160k_ip102-512x512', 'Segmenter ViT-B'),
    ('fcn_r50-d8_4xb4-160k_ip102-512x512', 'FCN ResNet-50'),
    ('pspnet_r50-d8_4xb4-160k_ip102-512x512', 'PSPNet ResNet-50'),
    ('deeplabv3plus_r50-d8_4xb4-160k_ip102-512x512', 'DeepLabV3+ ResNet-50'),
    ('segformer_mit-b2_8xb2-160k_ip102-512x512', 'SegFormer MiT-B2'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect per-model IP102 test results into summary files.')
    parser.add_argument(
        '--work-root',
        default='work_dirs/ip102',
        help='Root directory containing one work_dir per model.')
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to store summary.csv and summary.md.')
    return parser.parse_args()


def main():
    args = parse_args()
    work_root = Path(args.work_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for stem, display_name in MODEL_SPECS:
        metrics_path = work_root / stem / 'final_test_metrics.json'
        if not metrics_path.is_file():
            raise FileNotFoundError(f'Missing metrics file: {metrics_path}')
        with metrics_path.open('r', encoding='utf-8') as f:
            payload = json.load(f)

        foreground = payload['per_class']['foreground']
        summary = payload['summary']
        rows.append({
            'model': display_name,
            'IoU_fg': round(float(foreground['IoU']), 4),
            'Dice_fg': round(float(foreground['Dice']), 4),
            'Recall_fg': round(float(foreground['Recall']), 4),
            'mIoU': round(float(summary['mIoU']), 4),
            'mAcc': round(float(summary['mAcc']), 4),
            'best_ckpt': payload['checkpoint'],
            'work_dir': payload['work_dir'],
        })

    csv_path = output_dir / 'summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'model', 'IoU_fg', 'Dice_fg', 'Recall_fg', 'mIoU', 'mAcc',
                'best_ckpt', 'work_dir'
            ])
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / 'summary.md'
    header = (
        '| model | IoU_fg | Dice_fg | Recall_fg | mIoU | mAcc | best_ckpt |\n'
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |\n')
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['IoU_fg']:.4f} | {row['Dice_fg']:.4f} "
            f"| {row['Recall_fg']:.4f} | {row['mIoU']:.4f} "
            f"| {row['mAcc']:.4f} | `{row['best_ckpt']}` |\n")
    md_path.write_text(''.join(lines), encoding='utf-8')

    print(csv_path)
    print(md_path)


if __name__ == '__main__':
    main()
