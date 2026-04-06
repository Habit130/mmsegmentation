_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/plantseg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# Uniform-80k-samples profile: train batch 8,
# val/test batch 1 because PlantSeg validation images have mixed resolutions.
# max_iters 10000 so each model sees 80,000 crops.
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=10000,
        by_epoch=False,
    )
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=2500)
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = dict(batch_size=1)
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2500, save_best='mIoU', rule='greater'))
