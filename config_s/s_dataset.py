_samples_per_gpu = 8
_workers_per_gpu = 2

dataset_type = 'SDataset'

data_root = 'data/SDataset'

train_pipeline = [
    dict(type='LoadImageFromNpy'),
    dict(type='LoadAnnotationsFromNpy'),
    dict(type='CenterCropAndRescale', crop_size=(400, 400), scale=0.56, is_training_pipeline=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromNpy'),
    dict(type='CenterCropAndRescale', crop_size=(400, 400), scale=0.56, is_training_pipeline=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        transforms=[
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
