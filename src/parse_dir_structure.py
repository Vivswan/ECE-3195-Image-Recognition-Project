import os
import time

from src.path_functions import path_join


def parse_dir_structure(filepath, name):
    timestamp = str(int(time.time()))
    name = timestamp + "_" + ("" if name is None else name)

    if not os.path.exists(filepath):
        os.mkdir(filepath)
    if not os.path.exists(path_join(filepath, "models")):
        os.mkdir(path_join(filepath, "models"))
    if not os.path.exists(path_join(filepath, "tensorboard")):
        os.mkdir(path_join(filepath, "tensorboard"))

    models_path = path_join(filepath, f"models/{name}")
    tensorboard_path = path_join(filepath, f"tensorboard/{name}")

    os.mkdir(models_path)
    os.mkdir(tensorboard_path)
    return name, models_path, tensorboard_path
