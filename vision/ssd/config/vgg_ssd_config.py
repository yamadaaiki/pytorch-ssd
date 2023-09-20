import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


# image_size = 300

# for range 0 to 1
# image_size = 256
# image_mean = np.array([0, 0, 0])  # RGB layout
# image_std = 255.0

image_size = 256
image_mean = np.array([123, 117, 104])  # RGB layout
image_std = 1.0

iou_threshold = 0.2
center_variance = 0.1
size_variance = 0.2

# for input size 256 * 256 case1
# specs = [
#     SSDSpec(32, 8, SSDBoxSizes(3, 5), [2]),
#     SSDSpec(16, 16, SSDBoxSizes(5, 9), [2, 3]),
#     SSDSpec(8, 32, SSDBoxSizes(9, 17), [2, 3]),
#     SSDSpec(4, 64, SSDBoxSizes(17, 30), [2, 3]),
#     SSDSpec(2, 128, SSDBoxSizes(30, 52), [2]),
#     SSDSpec(1, 256, SSDBoxSizes(52, 77), [2])
# ]

# # for input size 256 * 256 case2
# specs = [
#     SSDSpec(32, 8, SSDBoxSizes(10, 20), [2]),
#     SSDSpec(16, 16, SSDBoxSizes(20, 40), [2, 3]),
#     SSDSpec(8, 32, SSDBoxSizes(40, 60), [2, 3]),
#     SSDSpec(4, 64, SSDBoxSizes(60, 80), [2, 3]),
#     SSDSpec(2, 128, SSDBoxSizes(80, 100), [2]),
#     SSDSpec(1, 256, SSDBoxSizes(100, 120), [2])
# ]

# for input size 256 * 256 case3
specs = [
    SSDSpec(32, 8, SSDBoxSizes(10, 30), [2]),
    SSDSpec(16, 16, SSDBoxSizes(30, 60), [2, 3]),
    SSDSpec(8, 32, SSDBoxSizes(60, 100), [2, 3]),
    SSDSpec(4, 64, SSDBoxSizes(100, 150), [2, 3]),
    SSDSpec(2, 128, SSDBoxSizes(150, 200), [2]),
    SSDSpec(1, 256, SSDBoxSizes(200, 260), [2])
]

# original
# specs = [
#     SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
#     SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
#     SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
#     SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
# ]


priors = generate_ssd_priors(specs, image_size)