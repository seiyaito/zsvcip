import argparse
import os
import sys

import torch
import torch.cuda
import torch.utils.data
from tqdm import tqdm
from zsvcip.config import get_default_config
from zsvcip.data import build_dataset
from zsvcip.evaluation import TextEvaluator
from zsvcip.modeling import build_model

argv = sys.argv
try:
    index = argv.index("--")
except ValueError:
    index = len(argv)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
parser.add_argument("-i", "--input")
parser.add_argument("-m", "--mode", default="text", choices=["text", "image"])
args = parser.parse_args(argv[1:index])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    cfg = get_default_config()

    if args.config is not None and os.path.exists(args.config):
        cfg.update_from_config_file(args.config)

    if index != len(argv):
        cfg.update_from_args(argv[index + 1 :])

    assert (
        cfg.model.arch == args.mode
    ), '"mode" must be the same as the model architecture.'
    model = build_model(cfg.model)
    model.to(device)

    if os.path.exists(cfg.resume):
        weights = torch.load(cfg.resume)
        try:
            model.load_state_dict(weights["model"])
        except:
            model.load_state_dict(weights["model"], strict=False)

    dataset = build_dataset(cfg.dataset)["test"]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model.eval()

    evaluators = [TextEvaluator()]

    for batch_data in tqdm(dataloader):
        text = batch_data[args.mode]
        targets = batch_data["target"]

        with torch.no_grad():
            output = model(text)
            preds = torch.sigmoid(output["logits"]).cpu()

        for evaluator in evaluators:
            evaluator.update(preds, targets)

    for evaluator in evaluators:
        print(evaluator.compute())


if __name__ == "__main__":
    evaluate()
