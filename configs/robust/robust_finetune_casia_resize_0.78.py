# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/casiav2_val.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py',
]
pretrained = 'work_dirs/hssn_soft_internimage_t_512_160k_manipulation_focal/best_mIoU_iter_160000.pth'
class_weight = [0.8, 1.2]
model = dict(
    type='BinarySegEncoderDecoder',
    init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
    ),
    neck=dict(
        type='SoftHSSN',
        spix_index=[0, 1, 2],
        in_channels=[64, 128, 256, 512],
        hidden_channels=20,
        out_channels=[64, 128, 256, 512],
        n_spix=[256, 128, 64],
        n_iter=5,
        pos_scale=2.5,
        color_scale=0.26,
        fuse_channels=256,
        fuse_resolution=(128, 128),
        freeze_kernel=True,
    ),
    decode_head=dict(
        num_classes=2,
        in_channels=[64, 128, 256, 512],
        ignore_index=127,
        loss_decode=dict(
            type='FocalLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            class_weight=class_weight
        )),
    auxiliary_head=dict(
        num_classes=2,
        in_channels=256,
        ignore_index=127,
        loss_decode=dict(
            type='FocalLoss',
            use_sigmoid=True,
            loss_weight=0.4,
            class_weight=class_weight
        )),
    test_cfg=dict(mode='whole')
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(2048, 512),
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='TestAug', aug_type='Resize', aug_arg=0.78),
            dict(type='Resize', keep_ratio=False),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=1.0,
                       depths=[4, 4, 18, 4]))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU

data = dict(samples_per_gpu=10,
            workers_per_gpu=14,
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))
runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=500, max_keep_ckpts=1)
evaluation = dict(interval=500, metric='mIoU', save_best='mIoU')
# fp16 = dict(loss_scale=dict(init_scale=512))
