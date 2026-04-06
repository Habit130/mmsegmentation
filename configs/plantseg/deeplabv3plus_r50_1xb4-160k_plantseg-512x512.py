_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/plantseg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# Uniform-80k profile: train batch 16, val/test batch 4.
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=2, norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=2, norm_cfg=norm_cfg),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

optim_wrapper = dict(optimizer=dict(lr=0.01))
param_scheduler = [dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=80000, by_epoch=False)]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
train_dataloader = dict(batch_size=16)
val_dataloader = dict(batch_size=4)
test_dataloader = dict(batch_size=4)
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=8000, save_best='mIoU', rule='greater'))
