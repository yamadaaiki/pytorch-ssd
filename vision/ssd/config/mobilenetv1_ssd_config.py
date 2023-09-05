import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 1
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, ]), # 50
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]), # 18 
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, ]), # 8
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, ]) # 2
]


priors = generate_ssd_priors(specs, image_size)