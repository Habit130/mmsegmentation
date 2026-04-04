_base_ = [
    '../segmenter/segmenter_vit-b_mask_8xb1-160k_ade20k-512x512.py',
    '../_base_/datasets/plantseg.py'
]

model = dict(
    decode_head=dict(num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

optimizer = dict(lr=0.000125, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = dict(batch_size=1)
default_hooks = dict(checkpoint=dict(save_best='mIoU', rule='greater'))
