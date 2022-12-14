from dataclasses import dataclass, field
import torch


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
