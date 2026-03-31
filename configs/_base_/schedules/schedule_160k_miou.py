_base_ = ['./schedule_160k.py']

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=16000,
        save_best='mIoU',
        rule='greater'))
