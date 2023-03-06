from mmseg.datasets.builder import PIPELINES
import mmcv
import numpy as np


@PIPELINES.register_module()
class CenterCropAndRescaleForAnnotations(object):
    def __init__(self, crop_size=(400, 400), scale=0.56):
        self.crop_size = crop_size
        self.scale = scale

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]

        original_img = results['gt_semantic_seg']
        img_height, img_width = original_img.shape[:2]

        x1 = int((img_width - crop_width) / 2)
        y1 = int((img_height - crop_height) / 2)
        x2 = x1 + crop_width - 1
        y2 = y1 + crop_height - 1

        cropped_img = mmcv.imcrop(original_img, bboxes=np.array([x1, y1, x2, y2]))
        results['gt_semantic_seg'] = cropped_img

        if self.scale != 1.0:
            results['gt_semantic_seg'] = mmcv.imrescale(results['gt_semantic_seg'], self.scale,
                                                        interpolation='nearest')

        return results
