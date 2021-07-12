import json
import re
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.summary import summary
from src.train_and_test import train, test


def arg_optimizer(optimizer: Callable, *args, **kwargs):
    def optimizer_fn(params):
        return optimizer(params, *args, **kwargs)

    return optimizer_fn


class NNModule(nn.Module):
    def __init__(self, log_dir: str, device: torch.device):
        super().__init__()

        self.optimizer: Optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.tb = SummaryWriter(log_dir=log_dir)
        self.device = device

    def compile(self, optimizer: Optimizer, loss_fn, scheduler=None):
        self.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.tb.add_text("summary", re.sub("\n", "\n    ", "    " + summary(self, (3, 32, 32), self.device)))
        self.tb.add_text("str", re.sub("\n", "\n    ", "    " + str(self)))

        self.tb.add_text("compile_parameters", re.sub("\n", "\n    ", "    " + json.dumps({
            "device": str(self.device),
            "optimizer": str(self.optimizer),
            "loss_fn": self.loss_fn.__name__,
            "scheduler": str(self.scheduler),
        }, sort_keys=True, indent=2)))

        self.tb.add_graph(self, torch.zeros((2, 3, 32, 32)).to(self.device))
        self._add_layer_data(epoch=-1)
        return self

    def _add_layer_data(self, epoch: int = None):
        idx = 0
        for module in self.modules():
            if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList) or (module == self):
                continue

            idx += 1
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                self.tb.add_histogram(f"{idx}-{module}.bias", module.bias, epoch)
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                self.tb.add_histogram(f"{idx}-{module}.weight", module.weight, epoch)

    def fit(self, train_loader: DataLoader, test_loader: DataLoader, epoch: int = None):
        train_loss, train_accuracy = train(self, self.device, train_loader, self.optimizer, self.loss_fn, epoch)
        test_loss, test_accuracy = test(self, self.device, test_loader, self.loss_fn)

        self.tb.add_scalar('Loss/train', train_loss, epoch)
        self.tb.add_scalar("Accuracy/train", train_accuracy, epoch)
        self.tb.add_scalar('Loss/test', test_loss, epoch)
        self.tb.add_scalar("Accuracy/test", test_accuracy, epoch)
        self._add_layer_data(epoch=epoch)

        if self.scheduler is not None:
            self.scheduler.step()

        return train_loss, train_accuracy, test_loss, test_accuracy

    def close(self):
        self.tb.close()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__()
        self.tb.close()
