from fourier_1d import FNO1d
from trainer import Trainner
from config import TrainConfig
from utilities3 import MatReader
import pathlib
from dataclasses import asdict

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)


DATA_DIR = pathlib.Path("./data/")
BURGERS_DATA_PATH = DATA_DIR / "burgers_data_R10.mat"
SAVE_PATH = pathlib.Path("./trained_model/burgers_equation_demo01.h")

traincfg = TrainConfig(20, 0.001, 100, 50, 0.5, "cpu")
dataloader = MatReader('data/burgers_data_R10.mat')
sub = 8
x_data = dataloader.read_field('a')[:, ::sub]
y_data = dataloader.read_field('u')[:, ::sub]


def main():
    print(traincfg)

    model = FNO1d(16, 64)
    trainer = Trainner(model, **asdict(traincfg))

    trainer.train(x_data, y_data, 0.8)
    trainer.save(SAVE_PATH)


if __name__ == "__main__":
    main()
