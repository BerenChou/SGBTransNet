from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class ToSingleChannel(object):
    def __call__(self, results):
        results['gt_semantic_seg'] = results['gt_semantic_seg'][:, :, 0] // 255
        return results
