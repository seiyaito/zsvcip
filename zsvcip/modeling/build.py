from .image_model import ImageModel
from .text_model import TextModel


def build_model(cfg):
    if cfg.arch == "text":
        return TextModel(**cfg)
    elif cfg.arch == "image":
        return ImageModel(**cfg)
    else:
        raise ValueError
