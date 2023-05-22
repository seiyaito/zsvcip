import argparse
import os.path
import sys

import torch
import torch.cuda
from PIL import Image
from zsvcip.config import get_default_config
from zsvcip.modeling import build_model

argv = sys.argv
try:
    index = argv.index("--")
except ValueError:
    index = len(argv)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
parser.add_argument("-i", "--input")
parser.add_argument("-m", "--mode", default="image", choices=["text", "image"])
args = parser.parse_args(argv[1:index])


def inference():
    cfg = get_default_config()

    if args.config is not None and os.path.exists(args.config):
        cfg.update_from_config_file(args.config)

    if index != len(argv):
        cfg.update_from_args(argv[index + 1 :])

    assert (
        cfg.model.arch == args.mode
    ), '"mode" must be the same as the model architecture.'
    model = build_model(cfg.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if os.path.exists(cfg.resume):
        weights = torch.load(cfg.resume)
        try:
            model.load_state_dict(weights["model"])
        except:
            model.load_state_dict(weights["model"], strict=False)

    if args.mode == "image":
        inputs = Image.open(args.input)
    else:
        inputs = args.input

    model.eval()
    output = model([inputs])
    prob = torch.sigmoid(output["logits"])[0].item()
    print(f"immorality: {prob}")


if __name__ == "__main__":
    inference()
