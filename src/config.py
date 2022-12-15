from dataclasses import dataclass, field
import torch


@dataclass(frozen=True)
class EQConfig:
    xmin: float = field(default=-5.)
    xmax: float = field(default=5.)
    samplesize: int = field(default=1024)
    noiserate: float = field(default=0.05)
    wavenoisescale: float = field(default=1.)
    dataset_samplesize: int = field(default=2000)
    tmax: float = field(default=0.1)

    def paramInitGene(self):
        return (self.xmin, self.xmax, self.samplesize)

    def paramgen(self):
        return (self.dataset_samplesize, self.tmax)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    step_size: int
    gamma: float
    device: str = field(default="cpu")

    def __post_init__(self):
        # tying check
        assert isinstance(self.batch_size, int)
        assert isinstance(self.learning_rate, float)
        assert isinstance(self.epochs, int)
        assert isinstance(self.device, str)
        assert isinstance(self.step_size, int)
        assert isinstance(self.gamma, float)

        if not torch.cuda.is_available() and self.device != "cpu":
            raise ValueError('gpu cant be used')
