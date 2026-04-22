_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/plantseg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_50e.py'
]

# Epoch-50 profile: one full pass over the PlantSeg train split per epoch.
crop_size = (512, 512)
work_dir = './work_dirs/plantseg_epoch50/pspnet_r50'
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=2, norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=2, norm_cfg=norm_cfg),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

optim_wrapper = dict(optimizer=dict(lr=0.01))
param_scheduler = [dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=50, by_epoch=True)]
train_dataloader = dict(batch_size=16)
val_dataloader = dict(batch_size=1)
test_dataloader = dict(batch_size=1)
default_hooks = dict(checkpoint=dict(by_epoch=True, interval=5, save_best='mIoU', rule='greater'))
