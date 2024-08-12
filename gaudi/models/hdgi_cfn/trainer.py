import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        data: Data,
        device: torch.device,
        lr: float,
        weight_decay: float,
        gamma: float,
        patience: int,
        epochs: int,
        checkpoints_path: str,
        seed: int = 1,
        **kwargs: Dict[str, Any],
    ):
        """
        Initialize the Trainer class.

        :param model: The model to train.
        :param data: The input data.
        :param device: The device to use for training (CPU or GPU).
        :param experiment_name: The name of the experiment.
        :param lr: Learning rate.
        :param weight_decay: Weight decay (L2 regularization).
        :param gamma: Multiplicative factor of learning rate decay.
        :param patience: Patience for learning rate scheduler.
        :param epochs: Number of training epochs.
        :param checkpoints_path: Path to save the model checkpoints.
        :param kwargs: Additional keyword arguments.
        """

        self.set_seed(seed)
        self.model = model.to(device)
        self.data = data
        self.device = device

        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=gamma, patience=patience, verbose=True
        )

        self.epochs = epochs
        self.checkpoints_path = checkpoints_path
        self.best_loss = float("inf")

    def _train_epoch(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        x = self.data.x.to(self.device)
        edge_index = self.data.edge_index.to(self.device)
        community_edge_index = self.data.community_edge_index.to(self.device)

        pos_z, neg_z, community_summary = self.model(
            x=x,
            edge_index=edge_index,
            community_edge_index=community_edge_index,
            separators=self.data.separators,
        )
        pos_node_z, pos_community_z = (
            pos_z[: self.data.separators[1]],
            pos_z[self.data.separators[-2] :],
        )
        neg_node_z, neg_community_z = (
            neg_z[: self.data.separators[1]],
            neg_z[self.data.separators[-2] :],
        )

        loss = self.model.loss(
            pos_node_z,
            neg_node_z,
            pos_community_z,
            neg_community_z,
            community_summary,
            tiles=self.data.primitives[self.data.separators[-2] :],
        )

        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss)
        return loss.item()

    def _save_best_model(self, loss: float) -> None:
        """Save the best model based on the current loss."""
        save_model_folder = Path(self.checkpoints_path)
        save_model_folder.mkdir(parents=True, exist_ok=True)

        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(self.model.state_dict(), save_model_folder / "best_model.pth")

    def train(self) -> None:
        """Train the model."""

        for epoch in tqdm(range(self.epochs)):

            loss = self._train_epoch()
            if epoch % 10 == 0:
                print(f'Loss: {loss:.4f}')

            self._save_best_model(loss)

    def get_embedding(self) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            x = self.data.x.to(self.device)
            edge_index = self.data.edge_index.to(self.device)

            community_edge_index = self.data.community_edge_index.to(self.device)

            (
                embeds,
                _,
                _,
            ) = self.model(
                x=x,
                edge_index=edge_index,
                community_edge_index=community_edge_index,
                separators=self.data.separators,
            )
            return embeds

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
