import argparse
import os.path
import sys

from zsvcip.config import get_default_config
from zsvcip.engine import Trainer

argv = sys.argv
try:
    index = argv.index("--")
except ValueError:
    index = len(argv)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
args = parser.parse_args(argv[1:index])


def main():
    cfg = get_default_config()
    if args.config is not None and os.path.exists(args.config):
        cfg.update_from_config_file(args.config)

    if index != len(argv):
        cfg.update_from_args(argv[index + 1 :])

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
