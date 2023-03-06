model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SGBTransNet',
        img_size=352,
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=2.0),
            dict(type='DiceLoss', loss_weight=3.0),
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
