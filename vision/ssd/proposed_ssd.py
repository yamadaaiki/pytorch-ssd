import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d, GroupNorm, ZeroPad2d
from ..nn.scaled_l2_norm import ScaledL2Norm
from ..nn.proposed_detect_metastases import prpposed_detect_metastases
import torch.nn as nn

from .ssd import SSD, Proposed_SSD
from .predictor import Predictor
from .config import proposed_ssd_config as config


def create_proposed_ssd(num_classes, is_test=False, in_channels=3):
    base_net = ModuleList([
        nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),])
    
    # これはなんだ？
    source_layer_indexes = [
        (10, ScaledL2Norm(128, 10)),
        (17, ScaledL2Norm(256, 5)),
        22,
        ]
    extras = ModuleList([
            ZeroPad2d(1),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                   padding='valid', stride=2),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(inplace=True),
            ZeroPad2d(1),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                   padding='valid', stride=2),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(inplace=True),
            ZeroPad2d(1),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                   padding='valid', stride=2),
            ReLU(inplace=True),])
    
    regression_headers = ModuleList([
        Conv2d(in_channels=128,  out_channels=2 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=2 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=2 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=512,  out_channels=2 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=2 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=2 * 4, kernel_size=3, padding=1), ]) # TODO: change to kernel_size=1, padding=0?
    
    classification_headers = ModuleList([
        Conv2d(in_channels=128,  out_channels=2 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=2 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=2 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512,  out_channels=2 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=2 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256,  out_channels=2 * num_classes, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
        ])
    
    return Proposed_SSD(num_classes, base_net, source_layer_indexes,
                        extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_proposed_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
