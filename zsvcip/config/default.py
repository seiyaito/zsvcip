from copy import deepcopy

from .config import CfgNode

DC = CfgNode()

# input
DC.INPUT = CfgNode()
DC.INPUT.BATCH_SIZE = 64


# dataset
DC.DATASET = CfgNode()
DC.DATASET.ROOT = "datasets"
DC.DATASET.NAME = "ethics"

DC.DATASET.ETHICS = CfgNode()
DC.DATASET.ETHICS.CATEGORY = "commonsense"
DC.DATASET.ETHICS.TRAIN_FILTERING = None
DC.DATASET.ETHICS.TEST_SPLIT = "test"
DC.DATASET.ETHICS.TEST_FILTERING = None

# model
DC.MODEL = CfgNode()
DC.MODEL.ARCH = "text"
DC.MODEL.CLIP_MODEL = "openai/clip-vit-base-patch32"
DC.MODEL.HIDDEN_DIM = 512
DC.MODEL.DROPOUT = 0.5
DC.MODEL.FREEZE_TEXT_ENCODER = True
DC.INPUT.TEXT_MAX_LENGTH = 77

# solver
DC.SOLVER = CfgNode()
DC.SOLVER.LR = 0.002
DC.SOLVER.WEIGHT_DECAY = 1e-2
DC.SOLVER.EPS = 1e-8
DC.SOLVER.EPOCHS = 100

# output
DC.OUTPUT = CfgNode()
DC.OUTPUT.PATH = "outputs"

# logging
DC.LOG = CfgNode()
DC.LOG.SNAPSHOT_STEP = 100
DC.LOG.PRINT_STEP = 100
DC.LOG.DEBUG = False

DC.SEED = 46
DC.RESUME = ""
DC.DEBUG = False


def get_default_config():
    return deepcopy(DC)
