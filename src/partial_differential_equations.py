import numpy as np
from abc import ABC, abstractmethod
from typing import List
from functools import partial
from dataclasses import dataclass


class InitFunction1D(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, x: float) -> float:
        """initial condition function.

        Args:
            x (float): arg

        Returns:
            float: result
        """
        pass


class RectangularFunction(InitFunction1D):
    def __init__(
            self, xmin: float = -1, xmax: float = 1, upper_value: float = 1, lowwer_value: float = 0) -> None:
        super().__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.upper_value = upper_value
        self.lower_value = lowwer_value

    def __call__(self, x: float) -> float:
        if self.xmin < x < self.xmax:
            return self.upper_value
        else:
            return self.lower_value


class Gaussian1D(InitFunction1D):

    def __init__(self, alpha: float = 1., center: float = 0.) -> None:
        super().__init__()
        self.alpha = alpha
        self.center = center

    def __call__(self, x: float) -> float:
        phi = x - self.center
        return np.exp(- phi * phi)


@dataclass()
class Condition1D:
    objects: List[float]
    xmin: float
    xmax: float
    variables: List[float]

    def __post_init__(self):
        assert all([isinstance(v, float) for v in self.objects])
        assert isinstance(self.xmin, float)
        assert isinstance(self.xmax, float)
        self.samplingsize = len(self.objects)
        self.samplef = (self.xmax - self.xmin) / self.samplingsize

    def __getitem__(self, x: float):
        ind = self.variables.index(x)
        return self.objects[ind]


class InitGenerator1D:
    def __init__(
            self, xmin: float, xmax: float, sampleing_size: int, noiserate: float = 0.05, wavenoisescale: float = 1.) -> None:

        self.xmin = xmin
        self.xmax = xmax
        self.sampling_size = sampleing_size
        self.X = [xmin + (i + 0.5) * ((self.xmax - self.xmin) / sampleing_size)
                  for i in range(self.sampling_size)]
        self.noiserate = noiserate
        self.wavenoisescale = wavenoisescale

        self.sinbaseset = [partial(self._sinnpi, n=i) for i in range(1, 5)]

    def generate_initcond(self, initfunc: InitFunction1D) -> Condition1D:
        ci = np.random.randn(4)
        self.ci = ci / np.linalg.norm(ci)
        objects = []
        for x in self.X:
            obs = initfunc(x) + self.gaussian_noise(self.noiserate,
                                                    0.1) + self.wavenoise(self.wavenoisescale, x)
            objects.append(obs)
        return Condition1D(objects, self.xmin, self.xmax, self.X)

    def gaussian_noise(self, samplingrate: float, scale: float) -> float:
        """add random gaussian noise 

        Args:
            samplingrate (float): rate for add noise
            scale (float): noise scale

        Returns:
            float: noise. 
        """        
        if np.random.rand() < samplingrate:
            noise = np.random.normal(0, scale)
        else:
            noise = 0
        return noise

    def wavenoise(self, scale: float, x: float) -> float:
        """generate wave noise. 

        Args:
            scale (float): wave scale
            x (float): target coordinate

        Returns:
            float: noise.
        """        
        noise = (1 / 5) * \
            sum(self.ci * np.array([func(x) for func in self.sinbaseset]))
        return scale * noise

    @staticmethod
    def _sinnpi(x: float, n: int) -> float:
        return np.sin(n * np.pi * x)


class HeatEquation1D:

    pi = np.pi

    def __init__(self, initcondition: Condition1D) -> None:

        self.initcond = initcondition
        self.xmin = self.initcond.xmin
        self.xmax = self.initcond.xmax
        self.samplef = self.initcond.samplef

        self.conv = self.initcond.variables  # convolutiona variables

    def u(self, x: float, t: float) -> float:
        """the general solution for this pde

        Args:
            x (float): coordinate for function
            t (float): time

        Returns:
            float: result in (x, t)
        """

        alpha = 1 / np.sqrt(4 * HeatEquation1D.pi * t)
        K = np.array(self.conv)
        Y = np.array(self.initcond.objects)

        delta = x - K
        delta2 = np.power(delta, 2)
        kernel = np.exp(- delta2 / (4 * t))  # heat kernel
        return alpha * np.sum(kernel * Y * self.samplef)

    def generate_lastcond(self, t: float) -> Condition1D:
        """generate condition in t=t

        Args:
            t (float): target time

        Returns:
            Condition1D: condition in t = t
        """
        objects = []
        for x in self.initcond.variables:
            objects.append(self.u(x, t))
        return Condition1D(objects, self.xmin, self.xmax, self.initcond.variables)