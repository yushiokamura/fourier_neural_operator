import random
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from partial_differential_equations import (
    InitGenerator1D,
    HeatEquation1D,
    Gaussian1D,
    RectangularFunction
)


@dataclass
class PdeDataset:
    X: np.array
    Y: np.array

    def save(self, savepath: str) -> None:
        """save dataset into numpy format.

        Args:
            savepath (str): save path
        """
        np.save(savepath, np.array([self.X, self.Y]))

    @staticmethod
    def load(path):
        XY = np.load(path)
        X = XY[0]
        Y = XY[1]

        return PdeDataset(X, Y)


class DataGenerator1D:
    def __init__(self, initgenerator: InitGenerator1D) -> None:
        self.initgene = initgenerator
        self._create_initfunctions()

    def generate(self, samplesize: int, t: float) -> PdeDataset:
        """generate pde dataset

        Args:
            samplesize (int): sample size of dataset
            t (float): target time.

        Returns:
            PdeDataset: dataset object. 
        """        

        U0 = []
        U_t = []
        for i in tqdm(range(samplesize)):

            func = random.choice(self.inifuncs)
            inicond = self.initgene.generate_initcond(func)
            eq = HeatEquation1D(inicond)
            lastcond = eq.generate_lastcond(t)

            U0.append(inicond.objects)
            U_t.append(lastcond.objects)

        return PdeDataset(np.array(U0), np.array(U_t))

    def _create_initfunctions(self):
        self.inifuncs = []
        for i in range(10):
            self.inifuncs.append(Gaussian1D(center=random.randrange(-2, 2)))
            self.inifuncs.append(
                RectangularFunction(-1, 1, 1 + 0.5 * random.random(), 0))
