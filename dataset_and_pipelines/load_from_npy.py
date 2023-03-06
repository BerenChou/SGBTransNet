import os.path as osp
import numpy as np
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromNpy(object):
    def __call__(self, results):
        filename = osp.join(results['img_prefix'], results['img_info']['filename'])

        img = np.load(filename)
        img = np.stack([img, img, img], 2)
        img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels, dtype=np.float32),
                                       std=np.ones(num_channels, dtype=np.float32),
                                       to_rgb=False)
        return results


@PIPELINES.register_module()
class LoadAnnotationsFromNpy(object):
    def __call__(self, results):
        filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])

        gt_semantic_seg = np.load(filename).astype(np.uint8)

        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results
