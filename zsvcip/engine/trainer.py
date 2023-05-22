import logging
import os
import random

import numpy as np
import torch.backends.cudnn
import torch.optim
import torch.utils.data
from zsvcip.data import build_dataset
from zsvcip.modeling import build_model
from zsvcip.utils import setup_logger


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        g = self.set_random_seed()

        self.model = build_model(cfg.model)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(cfg.solver.lr), eps=float(cfg.solver.eps)
        )

        self.start_epoch = 0
        if os.path.exists(self.cfg.resume):
            ckpt = torch.load(self.cfg.resume)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_epoch = ckpt["epoch"] if ckpt["epoch"] is not None else 0

        dataset = build_dataset(cfg.dataset)
        self.train_dataset, self.val_dataset = dataset["train"], dataset["val"]

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=cfg.input.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, pin_memory=True
        )

        os.makedirs(self.cfg.output.path, exist_ok=True)

        self.logger = setup_logger(
            __file__,
            level=logging.INFO if not self.cfg.debug else logging.DEBUG,
            output=os.path.join(cfg.output.path, "log.txt"),
        )

        self.logger.info(self.cfg)

    def train_one_epoch(self, epoch=0):
        self.model.train()
        for i, batch_data in enumerate(self.train_loader):
            global_step = i + epoch * len(self.train_loader)
            text = batch_data["text"]
            target = batch_data["target"].to(self.device)

            output = self.model(text, target)

            loss = output["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if global_step % self.cfg.LOG.PRINT_STEP == 0:
                self.logger.info(
                    f"Epoch: {epoch} global_step: {global_step} loss: {loss.item():.3f}"
                )

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.solver.epochs):
            self.train_one_epoch(epoch)
            if (epoch + 1) % self.cfg.log.snapshot_step == 0:
                self.save(f"epoch{epoch + 1:04d}", epoch=epoch + 1)
                self.save("latest", epoch=epoch + 1)

        self.save("latest", epoch=self.cfg.solver.epochs)

    def save(self, name, epoch=None):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            os.path.join(self.cfg.output.path, f"{name}.pth"),
        )

    def set_random_seed(self):
        seed = self.cfg.seed

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["PYTHONHASHSEED"] = "0"

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        g = None

        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        return g
