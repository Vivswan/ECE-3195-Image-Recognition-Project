import math

import torch
from torch import nn
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR

from src.NNModule import NNModule
from src.cuda import is_using_cuda
from src.load_cifar10 import load_cifar10
from src.loss_functions import nll_loss_fn
from src.parse_dir_structure import parse_dir_structure
from src.path_functions import path_join, get_relative_path
from src.summary import summary

DEVICE_NAME = None
DATA_FOLDER = get_relative_path(__file__, "data")
DRY_RUN = False
VERBOSE_LOG_FILE = False


class NNModel(NNModule):
    def __init__(self, out_features, device: torch.device, log_dir: str):
        super().__init__(log_dir, device)

        self.conv_nn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        self.flatten = nn.Sequential(
            nn.Flatten(start_dim=1)
        )

        self.fc_nn = nn.Sequential(
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, out_features),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_nn(x)
        x = self.flatten(x)
        x = self.fc_nn(x)
        return x


def assignment_p1(
        name="assignment_p1",
        batch_size=64,
        epochs=14,
        gamma=0.7,
        learning_rate=1,
        loss_fn=nll_loss_fn,
        device_name=DEVICE_NAME,
        data_folder=DATA_FOLDER,
):
    name, models_path, tensorboard_path = parse_dir_structure(get_relative_path(__file__, data_folder), name)
    device, is_cuda = is_using_cuda(device_name)
    log_file = path_join(data_folder, f"{name}_logs.txt")

    train_loader, test_loader, classes = load_cifar10(
        path=path_join(data_folder, "cifar10"),
        batch_size=batch_size,
        is_cuda=is_cuda
    )

    model = NNModel(
        out_features=len(classes),
        device=device,
        log_dir=tensorboard_path,
    )

    optimizer = Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    model.compile(
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn
    )

    if VERBOSE_LOG_FILE:
        with open(log_file, "a+") as file:
            file.write(str(model) + "\n\n")
            file.write(summary(model, (3, 32, 32), device) + "\n\n")

    for epoch in range(epochs):
        train_loss, train_accuracy, test_loss, test_accuracy = model.fit(train_loader, test_loader, epoch)

        str_epoch = str(epoch).zfill(math.ceil(math.log10(epochs)))
        print_str = f'({str_epoch})' \
                    f' Training Loss: {train_loss:.4f},' \
                    f' Training Accuracy: {100. * train_accuracy:.0f}%' \
                    f' Testing Loss: {test_loss:.4f},' \
                    f' Testing Accuracy: {100. * test_accuracy:.0f}%\n'

        print(print_str)
        if VERBOSE_LOG_FILE:
            with open(log_file, "a+") as file:
                file.write(print_str)

        if DRY_RUN:
            break

    model.close()


if __name__ == '__main__':
    assignment_p1()
