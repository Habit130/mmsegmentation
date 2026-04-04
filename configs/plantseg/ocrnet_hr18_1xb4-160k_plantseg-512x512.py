_base_ = [
    '../ocrnet/ocrnet_hr18_4xb4-160k_ade20k-512x512.py',
    '../_base_/datasets/plantseg.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[18, 36, 72, 144],
            channels=270,
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

optim_wrapper = dict(optimizer=dict(lr=0.0025))
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = dict(batch_size=1)
default_hooks = dict(checkpoint=dict(save_best='mIoU', rule='greater'))
