_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/plantseg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# Budget-640k benchmark profile: batch 2 x 320k iters = 640k crops.
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

optimizer = dict(lr=0.00025, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=320000,
        by_epoch=False,
    )
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=320000, val_interval=32000)
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = dict(batch_size=1)
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=32000, save_best='mIoU', rule='greater'))
