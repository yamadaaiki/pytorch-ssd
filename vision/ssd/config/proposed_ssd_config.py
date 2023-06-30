import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 256
# koreha nanini setteishitara iinoka?
image_mean = np.array([127, 127, 127]) # if use gray scale, only 1
image_std = 1.0

iou_threshold = 0.20
center_variance = 0.1
size_variance = 0.2

# SSDSpec = \
# collections.namedtuple('SSDSpec', 
# ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])
# SSDBoxSizes = \
# collections.namedtuple('SSDBoxSizes', ['min', 'max'])

# box size is [2, 4, 8, 16, 30, 52]
specs = [
    SSDSpec(128, 2, SSDBoxSizes(3,  62), [1]),
    SSDSpec(64,  4, SSDBoxSizes(5,  62), [1]),
    SSDSpec(32,  8, SSDBoxSizes(9,  62), [1]),
    SSDSpec(16, 16, SSDBoxSizes(17, 62), [1]),
    SSDSpec(8,  32, SSDBoxSizes(30, 62), [1]),
    SSDSpec(4,  64, SSDBoxSizes(52, 62), [1])
    ]


priors = generate_ssd_priors(specs, image_size)