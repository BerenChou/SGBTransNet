from mmseg.datasets.builder import PIPELINES
import mmcv
import numpy as np


@PIPELINES.register_module()
class CenterCropAndRescale(object):
    def __init__(self, crop_size=(400, 400), scale=0.56, is_training_pipeline=True):
        self.crop_size = crop_size
        self.scale = scale
        self.is_training_pipeline = is_training_pipeline

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]

        original_img = results['img']
        img_height, img_width = original_img.shape[:2]

        x1 = int((img_width - crop_width) / 2)
        y1 = int((img_height - crop_height) / 2)
        x2 = x1 + crop_width - 1
        y2 = y1 + crop_height - 1

        cropped_img = mmcv.imcrop(original_img, bboxes=np.array([x1, y1, x2, y2]))
        results['img'] = cropped_img

        if self.scale != 1.0:
            results['img'] = mmcv.imrescale(results['img'], self.scale)

        if self.is_training_pipeline:
            results['gt_semantic_seg'] = mmcv.imcrop(results['gt_semantic_seg'], bboxes=np.array([x1, y1, x2, y2]))
            if self.scale != 1.0:
                results['gt_semantic_seg'] = mmcv.imrescale(results['gt_semantic_seg'], self.scale,
                                                            interpolation='nearest')

        results['img_shape'] = results['img'].shape
        results['ori_shape'] = results['img'].shape
        results['pad_shape'] = results['img'].shape
        return results
