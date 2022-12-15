import json
import argparse
from fourier_1d import FNO1d
from trainer import Trainner
from partial_differential_equations import InitGenerator1D
from data_generator import DataGenerator1D
from config import TrainConfig, EQConfig
import pathlib
from dataclasses import asdict


import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("config")

if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.config) as f:
        cfgdict = json.load(f)

    DATASET_PATH = pathlib.Path(cfgdict["dataset_path"])
    SAVE_PATH = pathlib.Path(cfgdict["trained_model_path"])

    eqcfg = EQConfig(**cfgdict["eq"])
    traincfg = TrainConfig(**cfgdict["train"])

    inigene = InitGenerator1D(*eqcfg.paramInitGene())
    dg = DataGenerator1D(inigene)
    dataset = dg.generate(*eqcfg.paramgen())
    x_data = torch.Tensor(dataset.X)
    y_data = torch.Tensor(dataset.Y)
    dataset.save(DATASET_PATH)

    print(traincfg)

    model = FNO1d(16, 64)
    trainer = Trainner(model, **asdict(traincfg))
    trainer.train(x_data, y_data)
    trainer.save(SAVE_PATH)
