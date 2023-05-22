import torchmetrics


class TextEvaluator:
    def __init__(self, threshold=0.5):
        self.accuracy = torchmetrics.Accuracy(task="binary", threshold=threshold)

    def update(self, preds, targets):
        self.accuracy.update(preds, targets)

    def compute(self):
        return self.accuracy.compute()
