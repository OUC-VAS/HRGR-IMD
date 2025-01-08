from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from .manipulation import ManipulationDetectionDataset


@DATASETS.register_module()
class CASIAV2Dataset(ManipulationDetectionDataset):
    CLASSES = ('Real', 'Fake')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(CASIAV2Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_gt.png',
            **kwargs)
