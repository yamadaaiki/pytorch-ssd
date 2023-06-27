import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d, GroupNorm, ZeroPad2d
from ..nn.proposed_detect_metastases import prpposed_detect_metastases

from .ssd import SSD
from .predictor import Predictor
from .config import proposed_ssd_config as config


def create_proposed_ssd(num_classes, is_test=False):
    vgg_config = [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M']
    base_net = ModuleList(prpposed_detect_metastases(vgg_config))
    
    # これはなんだ？
    source_layer_indexes = [
        (23, GroupNorm(8, 512)),
        len(base_net),
        ]
    extras = ModuleList([
        Sequential(
            ZeroPad2d(1),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                   padding='valid', padding_mode='', stride=2),
            ReLU(),),
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),),
        Sequential(
            ZeroPad2d(1),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                   padding='valid', padding_mode='', stride=2),
            ReLU(),),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),),
        Sequential(
            ZeroPad2d(1),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                   padding='valid', padding_mode='', stride=2),
            ReLU(),),
        ])
    
    regression_headers = ModuleList([
        Conv2d(in_channels=128,  out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=512,  out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=4 * 4, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
        ])
    
    classification_headers = ModuleList([
        Conv2d(in_channels=128,  out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512,  out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=4 * num_classes, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
        ])
    
    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_proposed_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
