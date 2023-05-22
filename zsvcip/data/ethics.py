import os

import polars as pl
import torch.utils.data as data


class ETHICS(data.Dataset):
    AMBIGUOUS = -1
    IMMORAL = 0
    MORAL = 1

    def __init__(self, root, split="train", filtering=None):
        assert split in ["train", "test", "test_hard", "ambig"]
        self.root = root  # ethics/commonsense
        self.split = split

        path = os.path.join(self.root, f"cm_{self.split}.csv")
        df = pl.read_csv(path, has_header=(self.split != "ambig"))

        if filtering is not None and "is_short" not in df.columns:
            raise ValueError

        if filtering == "short":
            df = df.filter(pl.col("is_short"))

        if filtering == "long":
            df = df.filter(~pl.col("is_short"))

        if self.split == "ambig":
            self.target = [self.AMBIGUOUS] * df.select(pl.count()).item()
            self.text = df.to_numpy().ravel()
        else:
            self.target = df.select("label").to_numpy().ravel()
            self.text = df.select("input").to_numpy().ravel()

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return {"target": self.target[index], "text": self.text[index]}
