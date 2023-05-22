import torch.nn as nn
from transformers import AutoProcessor, CLIPModel


class ImageEncoder(nn.Module):
    def __init__(self, name=None, max_length=None):
        super().__init__()
        self.max_length = max_length
        if name is None:
            name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(name)
        self.processor = AutoProcessor.from_pretrained(name)
        self.projection_dim = self.model.projection_dim

    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt").to(self.model.device)
        image_features = self.model.get_image_features(**inputs)
        return image_features
