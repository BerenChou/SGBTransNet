_samples_per_gpu = 8
_workers_per_gpu = 4

dataset_type = 'CPSDataset'

data_root = 'data/CPSDataset'

img_norm_cfg = dict(mean=[126.461, 79.141, 57.480], std=[74.428, 53.394, 41.236], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ToSingleChannel'),
    dict(type='RandomFlip', prob=0.),
    dict(type='RandomRotate', prob=0., degree=360., seg_pad_val=0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(352, 352),
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=_samples_per_gpu,
    workers_per_gpu=_workers_per_gpu,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
)
