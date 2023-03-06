from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import numpy as np
import os.path as osp
from dataset_and_pipelines.center_crop_and_rescale_for_ann import CenterCropAndRescaleForAnnotations


@DATASETS.register_module()
class SDataset(CustomDataset):

    CLASSES = ('background', 'aorta', 'gallbladder', 'left_kidney', 'right_kidney',
               'liver', 'pancreas', 'spleen', 'stomach')

    PALETTE = [[0, 0, 0],
               [244, 35, 232],
               [128, 64, 128],
               [102, 102, 156],
               [190, 153, 153],
               [152, 251, 152],
               [250, 170, 30],
               [220, 220, 0],
               [107, 142, 35]]

    def __init__(self, **kwargs):
        super(SDataset, self).__init__(img_suffix='.npy', seg_map_suffix='.npy', **kwargs)
        self.center_crop_and_rescale_for_ann = CenterCropAndRescaleForAnnotations()

    def get_gt_seg_map_by_idx(self, index):
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        gt_seg_map = np.load(osp.join(results['seg_prefix'], results['ann_info']['seg_map'])).astype(np.uint8)
        results = {'gt_semantic_seg': gt_seg_map}
        self.center_crop_and_rescale_for_ann(results)
        return results['gt_semantic_seg']
