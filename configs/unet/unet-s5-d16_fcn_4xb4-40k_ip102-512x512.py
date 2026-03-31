_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/datasets/ip102_binary.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

model = dict(
    auxiliary_head=None,
    test_cfg=dict(mode='whole', _delete_=True))
