from collections import OrderedDict

import torch.nn as nn

from .classifier import Classifier
from .image_encoder import ImageEncoder


class ImageModel(nn.Module):
    def __init__(
        self,
        clip_model,
        hidden=512,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(clip_model)
        self.classifier = Classifier(self.image_encoder.projection_dim, hidden, dropout)

    def forward(self, x, target=None):
        out = OrderedDict()
        x = self.image_encoder(x)
        x = x.flatten(1)
        x = self.classifier(x)
        out["logits"] = x.squeeze(1)
        return out
