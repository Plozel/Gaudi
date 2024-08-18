import logging
from pathlib import Path
import torch
from torch_geometric.nn import GCNConv
import numpy as np

from .hdgi_cfn import layers
from .hdgi_cfn.model import HierarchicalDeepGraphInfomax
from .hdgi_cfn.trainer import Trainer


class RepresentationsGenerator:
    def __init__(
        self, gaudi_obj=None, sample=None, data=None, checkpoints_path=None, seed=1
    ):

        self.set_seed(seed)
        self.gaudi_obj = gaudi_obj
        if self.gaudi_obj is not None:
            self.data = self.gaudi_obj.multilevel_spatial_graph[0]
            self.sample = self.gaudi_obj.sample
            self.checkpoints_path = self.gaudi_obj.checkpoints_path
        else:
            self.data = data
            self.sample = sample
            self.checkpoints_path = checkpoints_path
        self.trainer = None

    def train_or_load_model(
        self,
        hidden_dim,
        output_dim,
        device,
        lr,
        weight_decay,
        gamma,
        patience,
        epochs,
        train,
        conv_layer,
        activation_fn,
        num_layers,
        seed,
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        device = torch.device(device)
        # Assuming that the first sample is representative of the dataset's initial feature dimension
        input_dim = self.data.x.shape[1]
        encoder = layers.CommunityFocusedNetwork(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, conv_layer=conv_layer, activation_fn=activation_fn, num_layers=num_layers)

        model = HierarchicalDeepGraphInfomax(
            output_dim=output_dim,
            encoder=encoder,
            summary=layers.summary,
            corruption=layers.corruption,
        )

        self.trainer = Trainer(
            model=model,
            data=self.data,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            gamma=gamma,
            patience=patience,
            epochs=epochs,
            checkpoints_path=self.checkpoints_path,
            seed=seed,
        )

        if train:
            self.trainer.train()
        else:
            logging.info("Loading a pre-trained model.")
            

        model_path = Path(self.checkpoints_path) / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                "A model was not found at the specified path. Please train a new model first."
            )
        
        model = HierarchicalDeepGraphInfomax(
            output_dim=output_dim,
            encoder=encoder,
            summary=layers.summary,
            corruption=layers.corruption,
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        self.trainer = Trainer(
            model=model,
            data=self.data,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            gamma=gamma,
            patience=patience,
            epochs=epochs,
            checkpoints_path=self.checkpoints_path,
            seed=seed,
        )

    def get_embedding_list(self):
        if self.trainer is None:
            raise RuntimeError(
                "No model is was trained or loaded. Please call train_or_load_model first."
            )
        separators = self.sample.separators
        embeds = self.trainer.get_embedding()
        embeds = embeds.detach().cpu().numpy()
        embeds_list = []
        for level in range(2):
            level_embeds = embeds[separators[level] : separators[level + 1]]
            embeds_list.append(level_embeds)

        return embeds_list

    def get_representations(self, train=False, hidden_dim=2000, output_dim=150, device=None, lr=1e-4, weight_decay=1e-5, gamma=0.1, patience=10, epochs=500, conv_layer=GCNConv, activation_fn=torch.nn.PReLU, num_layers=2, seed=1):
        """
        Generates cell and community level representations using the Heirarchical Deep Graph Infomax learning procedure integrated with the Community-Focused Networks architecture.
        """
        self.train_or_load_model(train=train, hidden_dim=hidden_dim, output_dim=output_dim, device=device, lr=lr, weight_decay=weight_decay, gamma=gamma, patience=patience, epochs=epochs, conv_layer=conv_layer, activation_fn=activation_fn, num_layers=num_layers, seed=seed)
        embeds = self.get_embedding_list(self.data.separators)

        if self.gaudi_obj is not None:
            self.gaudi_obj._update_embedding(embeds)

        return embeds

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
