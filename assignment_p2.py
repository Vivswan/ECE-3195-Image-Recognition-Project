import math

from torch.optim import SGD, Adam

from src.NNModule import arg_optimizer
from src.cuda import is_using_cuda
from src.load_cifar10 import load_cifar10
from src.loss_functions import cross_entropy_loss_fn, nll_loss_fn
from src.parse_dir_structure import parse_dir_structure
from src.path_functions import path_join, get_relative_path
from src.summary import summary
from src.vgg11 import VGG11Model

DEVICE_NAME = None
DATA_FOLDER = get_relative_path(__file__, "data")
DRY_RUN = False
VERBOSE_LOG_FILE = False

P2_PARAMETERS = {
    "assignment_p2_0": {"epochs": 14},
    "assignment_p2_1a": {},
    "assignment_p2_0_nll": {"loss_fn": nll_loss_fn},
    "assignment_p2_1b": {"batch_norm": True},
    "assignment_p2_1c": {"leaky_relu": True},
    "assignment_p2_1c_b": {"leaky_relu": True, "batch_norm": True},
    "assignment_p2_2a_4": {
        "optimizer": arg_optimizer(SGD, lr=2E-4),
        "leaky_relu": True,
    },
    "assignment_p2_2a_3": {
        "optimizer": arg_optimizer(SGD, lr=1E-3),
        "leaky_relu": True,
    },
    "assignment_p2_2a_4b": {
        "optimizer": arg_optimizer(SGD, lr=2E-4),
        "leaky_relu": True,
        "batch_norm": True,
    },
    "assignment_p2_2a_3b": {
        "optimizer": arg_optimizer(SGD, lr=1E-3),
        "leaky_relu": True,
        "batch_norm": True,
    },
    "assignment_p2_2b": {
        "batch_size": 256,
        "leaky_relu": True,
    },
    "assignment_p2_2b_b": {
        "batch_size": 256,
        "leaky_relu": True,
        "batch_norm": True,
    },
    "assignment_p2_2c": {
        "init_class": "xavier_uniform_",
        "leaky_relu": True,
    },
    "assignment_p2_2c_b": {
        "init_class": "xavier_uniform_",
        "leaky_relu": True,
        "batch_norm": True,
    },
    "assignment_p2_2d": {
        "dropout": False,
        "leaky_relu": True,
    },
    "assignment_p2_2d_b": {
        "dropout": False,
        "leaky_relu": True,
        "batch_norm": True,
    },
}
for i in P2_PARAMETERS:
    P2_PARAMETERS[i]["name"] = i


def assignment_p2(
        name,
        batch_size=64,
        batch_norm=False,
        leaky_relu=False,
        dropout=True,
        init_class="kaiming_normal_",
        loss_fn=cross_entropy_loss_fn,
        optimizer=arg_optimizer(Adam, lr=2E-4, betas=(0.5, 0.999)),
        epochs=100,
        device_name=DEVICE_NAME,
        data_folder=DATA_FOLDER,
        kwargs=None
):
    name, models_path, tensorboard_path = parse_dir_structure(data_folder, name)
    device, is_cuda = is_using_cuda(device_name)
    log_file = path_join(data_folder, f"{name}_logs.txt")

    train_loader, test_loader, classes = load_cifar10(
        path=path_join(data_folder, "cifar10"),
        batch_size=batch_size,
        is_cuda=is_cuda
    )

    model = VGG11Model(
        out_features=len(classes),
        device=device,
        log_dir=tensorboard_path,
        batch_norm=batch_norm,
        leaky_relu=leaky_relu,
        dropout=dropout,
        init_class=init_class,
    )
    model.compile(
        optimizer=optimizer(model.parameters()),
        loss_fn=loss_fn
    )

    if VERBOSE_LOG_FILE:
        with open(log_file, "a+") as file:
            kwargs["optimizer"] = model.optimizer
            file.write(str(kwargs) + "\n\n")
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
    for i in P2_PARAMETERS:
        print(f"Running {i}")
        assignment_p2(kwargs=P2_PARAMETERS[i], **P2_PARAMETERS[i])
