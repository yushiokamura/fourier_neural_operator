from typing import Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.utilities3 import LpLoss


def create_train_test_loader(
        x_data,
        y_data,
        train_rate,
        batch_size) -> Tuple[DataLoader, DataLoader]:

    ntrain = int(len(x_data) * train_rate)
    ntest = len(x_data) - ntrain

    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    x_train = x_train.reshape(ntrain, -1, 1)
    x_test = x_test.reshape(ntest, -1, 1)

    trainset = TensorDataset(x_train, y_train)
    testset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


class Trainner:
    def __init__(
            self,
            model: torch.nn.Module,
            batch_size: int,
            learning_rate: float,
            epochs: int,
            device: str,
            step_size, gamma) -> None:

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = model
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)
        self.loss = LpLoss(size_average=False)
        self.trainlog = []

    def train(
            self, x_data: torch.Tensor, y_data: torch.Tensor, train_rate: float) -> None:
        """
        """

        train_loader, test_loader = create_train_test_loader(
            x_data, y_data, train_rate, self.batch_size)

        for ep in range(self.epochs):
            train_mse, train_l2 = self._train_one_epoch(train_loader)
            test_mse, test_l2 = self._validate_one_epoch(test_loader)
            self.scheduler.step()

            tlog = {"ep": ep, "train_mse": train_mse, "train_ls": train_l2,
                    "test_mse": test_mse, "test_ls": test_l2}
            self.trainlog.append(tlog)

            print(
                f"epoch: {ep+1} / {self.epochs},"
                f"train mse: {train_mse:.5g}, test_mse: {test_mse:.5g}")

    def _train_one_epoch(
            self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        """

        self.model.train()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)

            mse = F.mse_loss(out.view(self.batch_size, -1),
                             y.view(self.batch_size, -1), reduction='mean')
            l2 = self.loss(out.view(self.batch_size, -1),
                           y.view(self.batch_size, -1))
            l2.backward()

            self.optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        return train_mse, train_l2

    def _validate_one_epoch(
            self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        """

        self.model.eval()
        test_mse = 0.0
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x)
                mse = F.mse_loss(out.view(self.batch_size, -1),
                                 y.view(self.batch_size, -1), reduction='mean')
                test_l2 += self.loss(out.view(self.batch_size, -1),
                                     y.view(self.batch_size, -1)).item()
                test_mse += mse.item()

            test_mse / len(test_loader)
        return test_mse, test_l2

    def save(self, save_path: Path) -> None:
        """
        """

        torch.save(self.model.state_dict(), save_path)
