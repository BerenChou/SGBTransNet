from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class CPSDataset(CustomDataset):

    CLASSES = ('background', 'colon_polyp')
    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(CPSDataset, self).__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        results['gt_semantic_seg'] = results['gt_semantic_seg'][:, :, 0] // 255
        return results['gt_semantic_seg']
