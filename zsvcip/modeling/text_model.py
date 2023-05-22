from collections import OrderedDict

import torch.nn as nn

from .classifier import Classifier
from .text_encoder import TextEncoder


class TextModel(nn.Module):
    def __init__(
        self,
        clip_model,
        hidden=512,
        dropout=0.5,
        freeze_text_encoder=True,
        max_length=None,
        **kwargs,
    ):
        super().__init__()
        self.freeze_text_encoder = freeze_text_encoder

        self.text_encoder = TextEncoder(clip_model, max_length)
        self.classifier = Classifier(self.text_encoder.projection_dim, hidden, dropout)
        self.criteria = nn.BCEWithLogitsLoss()

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def forward(self, x, targets=None):
        out = OrderedDict()
        x = self.text_encoder(x)
        x = x.flatten(1)
        x = self.classifier(x)
        out["logits"] = x.squeeze(1)

        if targets is not None:
            out["loss"] = self.compute_loss(out["logits"], targets.float())

        return out

    def train(self, mode=True):
        super().train(mode=mode)
        if self.freeze_text_encoder:
            self.text_encoder.eval()

    def compute_loss(self, preds, targets):
        return self.criteria(preds, targets)
