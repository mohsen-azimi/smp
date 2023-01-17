import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm import tqdm

# !pip install -q segmentation-models-pytorch
# !pip install -q torchsummary

from torchsummary import summary
import segmentation_models_pytorch as smp



def return_models():

    n_classes = 2
    en = 'mobilenet_v2'  # encoder_name : resnet34
    ew = 'imagenet'  # encoder_weights
    ed = 5  # encoder_depth
    eos = 16  # encoder_output_stride

    models = {}
    # UNet
    # model = smp.Unet(encoder_name=encoder_name, encoder_weights=ew, classes=n_classes, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    model = smp.Unet(encoder_name=en, encoder_depth=ed, encoder_weights=ew, decoder_use_batchnorm=True,
                     decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, in_channels=3,
                     classes=n_classes,
                     activation=None, aux_params=None)
    models[model.__class__.__name__] = model
    # Unet++
    model = smp.UnetPlusPlus(encoder_name=en, encoder_depth=ed, encoder_weights=ew, decoder_use_batchnorm=True,
                             decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, in_channels=3,
                             classes=n_classes, activation=None, aux_params=None)
    models[model.__class__.__name__] = model

    # MAnet
    model = smp.MAnet(encoder_name=en, encoder_depth=ed, encoder_weights=ew, decoder_use_batchnorm=True,
                      decoder_channels=(256, 128, 64, 32, 16), decoder_pab_channels=64, in_channels=3,
                      classes=n_classes,
                      activation=None, aux_params=None)
    models[model.__class__.__name__] = model

    # Linknet
    model = smp.Linknet(encoder_name=en, encoder_depth=ed, encoder_weights=ew, decoder_use_batchnorm=True,
                        in_channels=3,
                        classes=n_classes, activation=None, aux_params=None)
    models[model.__class__.__name__] = model

    # FPN
    # model = smp.FPN(encoder_name=encoder_name, encoder_weights=encoder_weights,  classes=n_classes, activation=None,)
    model = smp.FPN(encoder_name=en, encoder_depth=ed, encoder_weights=ew, decoder_pyramid_channels=256,
                    decoder_segmentation_channels=128, decoder_merge_policy='add', decoder_dropout=0.2, in_channels=3,
                    classes=n_classes, activation=None, upsampling=4, aux_params=None)
    models[model.__class__.__name__] = model

    # # PSPNet
    model = smp.PSPNet(encoder_name=en, encoder_weights=ew, encoder_depth=ed, psp_out_channels=512,
                       psp_use_batchnorm=True,
                       psp_dropout=0.2, in_channels=3, classes=n_classes, activation=None, upsampling=8,
                       aux_params=None)
    models[model.__class__.__name__] = model
    # PAN
    model = smp.PAN(encoder_name=en, encoder_weights=ew, encoder_output_stride=eos, decoder_channels=32, in_channels=3,
                    classes=n_classes, activation=None, upsampling=4, aux_params=None)

    models[model.__class__.__name__] = model

    # DeepLabV3
    model = smp.DeepLabV3(encoder_name=en, encoder_depth=ed, encoder_weights=ew, decoder_channels=256, in_channels=3,
                          classes=n_classes, activation=None, upsampling=8, aux_params=None)
    models[model.__class__.__name__] = model

    # DeepLabV3+
    model = smp.DeepLabV3Plus(encoder_name=en, encoder_depth=ed, encoder_weights=ew, encoder_output_stride=eos,
                              decoder_channels=256, decoder_atrous_rates=(12, 24, 36), in_channels=3, classes=n_classes,
                              activation=None, upsampling=4, aux_params=None)
    models[model.__class__.__name__] = model

    return models
