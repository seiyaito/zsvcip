import torch.nn as nn
from transformers import AutoTokenizer, CLIPModel


class TextEncoder(nn.Module):
    def __init__(self, name=None, max_length=None):
        super().__init__()
        self.max_length = max_length
        if name is None:
            name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.projection_dim = self.model.projection_dim

    def forward(self, x):
        inputs = self.tokenizer(
            x,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).to(self.model.device)

        text_features = self.model.get_text_features(**inputs)

        return text_features
