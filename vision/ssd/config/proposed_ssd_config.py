import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = 256
image_mean = np.array([127, 127, 127]) # RGB layout
image_std = 1.0

iou_threshold = 0.2
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(128, 2, SSDBoxSizes(10, 30), [2]),
    SSDSpec(64, 4, SSDBoxSizes(30, 60), [2, 3]),
    SSDSpec(32, 8, SSDBoxSizes(60, 100), [2, 3]),
    SSDSpec(16, 16, SSDBoxSizes(100, 150), [2, 3]),
    SSDSpec(8, 32, SSDBoxSizes(150, 200), [2]),
    SSDSpec(4, 64, SSDBoxSizes(200, 260), [2])
]

priors = generate_ssd_priors(specs, image_size)