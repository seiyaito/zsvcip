import os.path


def build_dataset(cfg):
    name = cfg.name.lower()
    root = os.path.join(cfg.root, name)
    if name == "ethics":
        from .ethics import ETHICS

        train_filtering = cfg.ethics.train_filtering
        test_split = cfg.ethics.test_split
        cat = cfg.ethics.category
        test_filtering = cfg.ethics.test_filtering

        dataset = {
            "train": ETHICS(
                os.path.join(root, cat), split="train", filtering=train_filtering
            ),
            "val": ETHICS(
                os.path.join(root, cat), split=test_split, filtering=test_filtering
            ),
            "test": ETHICS(
                os.path.join(root, cat), split=test_split, filtering=test_filtering
            ),
        }

    else:
        raise ValueError

    return dataset
