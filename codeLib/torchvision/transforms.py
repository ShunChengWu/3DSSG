#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:03:12 2022

@author: sc
"""
import torch
from torchvision import transforms

class TrivialAugmentWide(transforms.TrivialAugmentWide):
     # def _augmentation_space(self, num_bins: int):
     #    return {
     #        # op_name: (magnitudes, signed)
     #        "Identity": (torch.tensor(0.0), False),
     #        "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
     #        "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
     #        "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
     #        "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
     #        "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
     #        "Brightness": (torch.linspace(0.0, 0.1, num_bins), True),
     #        "Color": (torch.linspace(0.0, 0.1, num_bins), True),
     #        "Contrast": (torch.linspace(0.0, 0.1, num_bins), True),
     #        "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
     #        "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
     #        # "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
     #        "AutoContrast": (torch.tensor(0.0), False),
     #        # "Equalize": (torch.tensor(0.0), False),
     #    }
    def _augmentation_space(self, num_bins: int):
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.2, 0.8, num_bins), True),
            "Color": (torch.linspace(0.2, 0.8, num_bins), True),
            "Contrast": (torch.linspace(0.2, 0.8, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.8, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            # "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            # "Equalize": (torch.tensor(0.0), False),
        }
class RandAugment(transforms.RandAugment):
      def _augmentation_space(self, num_bins, image_size):
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            # "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            # "Equalize": (torch.tensor(0.0), False),
        }
