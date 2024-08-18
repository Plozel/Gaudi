import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import reset, uniform


EPS = 1e-15


class HierarchicalDeepGraphInfomax(torch.nn.Module):
    """
    Hierarchical Deep Graph Infomax model.
    This class extends the Deep Graph Infomax frameowrk to consider both node and community level information.

    Args:
        output_dim (int): The summary space dimensionality.
        encoder (torch.nn.Module): The encoder module.
        summary (callable): The readout function.
        corruption (callable): The corruption function.
    """

    def __init__(
        self,
        output_dim: int,
        encoder: torch.nn.Module,
        summary: callable,
        corruption: callable,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        # Separate weight matrices for nodes and communities
        self.node_weight = Parameter(torch.empty(output_dim, output_dim))
        self.community_weight = Parameter(torch.empty(output_dim, output_dim))
        self.joint_weight = Parameter(torch.empty(output_dim, 2 * output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        # Xavier Initialization for node and community weights
        init.xavier_uniform_(self.node_weight)
        init.xavier_uniform_(self.community_weight)
        # He Initialization for joint weights
        init.kaiming_uniform_(self.joint_weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, *args, separators, **kwargs) -> tuple:
        pos_z = self.encoder(*args, **kwargs)
        pos_node_z, pos_community_z = pos_z[: separators[1]], pos_z[separators[-2] :]
        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor,)
        cor_args = cor[: len(args)]
        cor_kwargs = {key: value for key, value in zip(kwargs.keys(), cor[len(args) :])}
        neg_z = self.encoder(*cor_args, **cor_kwargs)
        node_summary, community_summary = self.summary(
            pos_node_z, pos_community_z, *args, **kwargs
        )
        return pos_z, neg_z, community_summary

    def discriminate(
        self, z: Tensor, summary: Tensor, level: str
    ) -> Tensor:
        """Discriminate function to handle both nodes and communities."""
        assert level in [
            "node",
            "community",
            "joint",
        ], "Level must be either 'node' or 'community'"

        if level == "node":
            weight = self.node_weight
        elif level == "community":
            weight = self.community_weight
        else:
            weight = self.joint_weight

        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(weight, summary))
        return torch.sigmoid(value)

    def loss(
        self,
        pos_node_z: Tensor,
        neg_node_z: Tensor,
        pos_community_z: Tensor,
        neg_community_z: Tensor,
        community_summary: Tensor,
        tiles: list[frozenset],
    ) -> Tensor:
        """Loss function that computes the total loss based on node and community information."""

        pos_joint_loss = 0
        neg_joint_loss = 0
        pos_node_loss = 0
        neg_node_loss = 0
        for tile_idx, tile in enumerate(tiles):
            tile = list(tile)
            pos_node_loss += -torch.log(
                self.discriminate(
                    pos_node_z[tile], pos_community_z[tile_idx], "node"
                )
                + EPS
            ).mean()
            neg_node_loss += -torch.log(
                1
                - self.discriminate(
                    neg_node_z[tile], pos_community_z[tile_idx], "node"
                )
                + EPS
            ).mean()

            expanded_community_summary = community_summary.unsqueeze(0).expand(
                pos_node_z[tile].size(0), -1
            )
            joint_summary = torch.cat(
                [pos_node_z[tile], expanded_community_summary], dim=1
            )

            pos_joint_loss += -torch.log(
                1 - self.discriminate(
                    pos_community_z[tile_idx], joint_summary, "joint"
                )
                + EPS
            ).mean()
            neg_joint_loss += -torch.log(
                self.discriminate(
                    neg_community_z[tile_idx], joint_summary, "joint"
                )
                + EPS
            ).mean()

        node_loss = (pos_node_loss + neg_node_loss) / len(tiles)
        joint_loss = (pos_joint_loss + neg_joint_loss) / len(tiles)

        pos_community_loss = -torch.log(
            self.discriminate(
                pos_community_z, community_summary, "community"
            )
            + EPS
        ).mean()
        neg_community_loss = -torch.log(1 - 
            self.discriminate(
                neg_community_z, community_summary, "community"
            )
            + EPS
        ).mean()

        community_loss = pos_community_loss + neg_community_loss

        total_loss = node_loss + community_loss + joint_loss
        return total_loss

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.output_dim})"
