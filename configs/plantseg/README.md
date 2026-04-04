# PlantSeg single-GPU configs

This directory contains the Linux-server delivery configs for the sibling
`../plantseg` dataset.

All configs share the same assumptions:

- binary segmentation with classes `background` and `foreground`
- `../plantseg/main.json` is the manifest source of truth
- training uses 512x512 crops
- validation and test use sliding-window inference
- best checkpoint selection is based on `mIoU`

Config list:

- `upernet_swin-t_1xb2-160k_plantseg-512x512.py`
- `ocrnet_hr18_1xb4-160k_plantseg-512x512.py`
- `fcn_hr18_1xb4-160k_plantseg-512x512.py`
- `segmenter_vit-b_1xb1-160k_plantseg-512x512.py`
- `fcn_r50_1xb4-160k_plantseg-512x512.py`
- `pspnet_r50_1xb4-160k_plantseg-512x512.py`
- `deeplabv3plus_r50_1xb4-160k_plantseg-512x512.py`
- `segformer_mit-b2_1xb2-160k_plantseg-512x512.py`
