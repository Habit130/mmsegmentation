# PlantSeg Linux Server Delivery

This delivery targets:

- Linux
- single RTX 4090
- CUDA 11.8
- Python 3.10
- sibling dataset directory `../plantseg`

## Environment

1. Create the conda environment from `environment.server.cuda118.py310.yml`.
2. Activate the environment.
3. If the environment already exists and currently has NumPy 2.x installed,
   downgrade it to `numpy<2` before running MMSeg.
4. Install `mmcv==2.1.0` with `mim`.
5. Install this repository in editable mode.

If `mim install mmcv==2.1.0` falls back to a source build, keep the same
version and complete the build instead of downgrading CUDA or Python.

## Dataset contract

The repository reads the original dataset directly from:

- `../plantseg/main.json`
- `../plantseg/images/`
- `../plantseg/ann/`

No dataset restructuring is required.

## Metrics

Formal validation uses `PlantSegMetric` and reports:

- `IoU` for the foreground class
- `Dice` for the foreground class
- `Recall` for the foreground class
- `mIoU` averaged over background and foreground
- `mACC` averaged over background and foreground

## Configs

The delivered single-GPU configs live in `configs/plantseg/`.

Recommended starting points:

- `configs/plantseg/upernet_swin-t_1xb2-160k_plantseg-512x512.py`
- `configs/plantseg/ocrnet_hr18_1xb4-160k_plantseg-512x512.py`
- `configs/plantseg/fcn_hr18_1xb4-160k_plantseg-512x512.py`
- `configs/plantseg/segmenter_vit-b_1xb1-160k_plantseg-512x512.py`
- `configs/plantseg/fcn_r50_1xb4-160k_plantseg-512x512.py`
- `configs/plantseg/pspnet_r50_1xb4-160k_plantseg-512x512.py`
- `configs/plantseg/deeplabv3plus_r50_1xb4-160k_plantseg-512x512.py`
- `configs/plantseg/segformer_mit-b2_1xb2-160k_plantseg-512x512.py`

All configs:

- train with 512x512 crops
- validate and test with sliding-window inference
- save the best checkpoint according to `mIoU`

Current benchmark profile is the locked "uniform-80k-samples" setting:

- `FCN / PSPNet / HRNet / OCRNet`: train batch 16, val/test batch 1, 5k iters
- `DeepLabV3+`: train batch 8, val/test batch 1, 10k iters
- `UperNet-Swin-T / SegFormer-B2`: train batch 16, val/test batch 1, 5k iters
- `Segmenter-ViT-B`: train batch 8, val/test batch 1, 10k iters
- all models see the same total training budget: `80,000` crops
- validation/checkpoint intervals are also aligned by sample budget rather than raw iter count

Note:

- some config filenames still contain legacy suffixes such as `1xb4-160k`
  for continuity with earlier delivery steps
- use the actual values inside the config file as the source of truth
- PlantSeg validation/test images are not guaranteed to share one resolution,
  so `val/test_dataloader.batch_size` must stay at `1` unless the evaluation
  pipeline is changed to enforce a common size

## Final command surface

Environment setup:

```bash
conda env create -f environment.server.cuda118.py310.yml
conda activate mmseg-plantseg-cu118
pip install "numpy<2"
mim install mmcv==2.1.0
pip install -e .
```

Train:

```bash
python tools/train.py configs/plantseg/<config-name>.py
```

Validate or test:

```bash
python tools/test.py configs/plantseg/<config-name>.py <checkpoint>
```

Batch training and mask export:

```bash
bash tools/plantseg_batch_runner.sh
```

Only run selected models:

```bash
bash tools/plantseg_batch_runner.sh fcn_r50 pspnet_r50 segformer_b2
```

Export both test and val masks:

```bash
bash tools/plantseg_batch_runner.sh --export-val
```

Default mask export location:

- `work_dirs/plantseg_batch/masks/<model_alias>/test/*.png`
- `work_dirs/plantseg_batch/masks/<model_alias>/val/*.png` when `--export-val` is enabled
