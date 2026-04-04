_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py',
    '../_base_/datasets/plantseg.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(num_classes=2, norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=2, norm_cfg=norm_cfg),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

optim_wrapper = dict(
    optimizer=dict(lr=0.0000075),
)

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = dict(batch_size=1)
default_hooks = dict(checkpoint=dict(save_best='mIoU', rule='greater'))
