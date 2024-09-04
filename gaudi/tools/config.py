import sys
import json
import argparse
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional


@dataclass
class DetectionConfig:
    min_samples: int
    min_cluster_size: int
    xi: float
    max_eps: int

@dataclass
class Config:
    adata_path: Optional[str] = None
    pca_components: Optional[int] = None
    min_genes: Optional[int] = None
    min_cells: Optional[int] = None
    min_counts: Optional[int] = None
    max_pct_mt: Optional[int] = None
    n_top_genes: Optional[int] = None
    sample_name: Optional[str] = None
    dataset_name: Optional[str] = None
    experiment_name: Optional[str] = None
    lr_db_path: Optional[str] = None
    geneset_db_path: Optional[str] = None
    cellchatdb_type: Optional[str] = None
    seed: Optional[int] = None
    community_detection_method: Optional[str] = None
    n_iter: Optional[int] = None
    first_step_detection_config: Optional[DetectionConfig] = None
    rest_steps_detection_config: Optional[DetectionConfig] = None
    max_edges_length: Optional[int] = None
    k_nearest_neighbors: Optional[int] = None
    device: Optional[str] = None
    train: Optional[bool] = None
    epochs: Optional[int] = None
    hidden_dim: Optional[int] = None
    output_dim: Optional[int] = None
    measures_to_calculate: Optional[List[str]] = None
    main_composition_key: Optional[str] = None

class GaudiArgumentParser:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.parser = argparse.ArgumentParser(description="Process command line arguments and config file.")
        self._setup_arguments()

    def _setup_arguments(self):
        self.parser.add_argument('--adata_path', type=str, help="Path to the AnnData file")
        self.parser.add_argument('--pca_components', type=int, help="Number of PCA components")
        self.parser.add_argument('--min_genes', type=int, help="Minimum number of genes")
        self.parser.add_argument('--min_counts', type=int, help="Minimum number of counts")
        self.parser.add_argument('--max_pct_mt', type=int, help="Maximum percentage of MT genes")
        self.parser.add_argument('--min_cells', type=int, help="Minimum number of cells")
        self.parser.add_argument('--n_top_genes', type=int, help="Number of top genes")
        self.parser.add_argument('--sample_name', type=str, help="Name of the sample")
        self.parser.add_argument('--experiment_name', type=str, help="Name of the experiment")
        self.parser.add_argument('--dataset_name', type=str, help="Name of the dataset")
        
        self.parser.add_argument('--lr_db_path', type=str, help="Path to the ligand-receptor database file")
        self.parser.add_argument('--geneset_db_path', type=str, help="Path to the gene set database file")
        self.parser.add_argument('--cellchatdb_type', type=str, help="Type of CellChat database (e.g., 'Mouse', 'Human')")
        
        self.parser.add_argument('--seed', type=int, help="Seed for random number generation to ensure reproducibility")
        self.parser.add_argument('--community_detection_method', type=str, help="Algorithm used for community detection")
        self.parser.add_argument('--n_iter', type=int, help="Number of iterations for the algorithm")
        self.parser.add_argument('--max_edges_length', type=int, help="Length of the edges in the graph")
        self.parser.add_argument('--k_nearest_neighbors', type=int, help="Number of nearest neighbors to consider in the graph")
        
        self.parser.add_argument('--device', type=str, help="Computational device to use (e.g., 'cuda:0', 'cpu')")
        self.parser.add_argument('--train', type=bool, help="Flag to determine if the model should be trained")
        self.parser.add_argument('--epochs', type=int, help="Number of epochs for training")
        self.parser.add_argument('--hidden_dim', type=int, help="Number of hidden channels in the neural network")
        self.parser.add_argument('--output_dim', type=int, help="Dimensionality of community embeddings")
        
        self.parser.add_argument('--measures_to_calculate', type=str, nargs='+', help="List of measures to calculate (e.g., 'mean', 'max')")
        self.parser.add_argument('--main_composition_key', type=str, help="the key of the main group composition")

    def _load_config_file(self) -> Dict[str, Any]:
        if not self.config_path:
            return {}
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Configuration file {self.config_path} not found. Using defaults.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON from {self.config_path}: {e}")
            return {}

    def parse_args(self, known_only=False) -> Config:
        if 'ipykernel' in sys.argv[0]:
            args = argparse.Namespace(config_path=self.config_path)
        elif known_only:
            args, unknown = self.parser.parse_known_args()
        else:
            args = self.parser.parse_args()

        config_data = self._load_config_file()

        # Merge command line arguments
        for key, value in vars(args).items():
            if value is not None:
                config_data[key] = value

        # Convert nested configurations
        if 'first_step_detection_config' in config_data:
            config_data['first_step_detection_config'] = DetectionConfig(**config_data['first_step_detection_config'])
        if 'rest_steps_detection_config' in config_data:
            config_data['rest_steps_detection_config'] = DetectionConfig(**config_data['rest_steps_detection_config'])

        # Convert list if it's a string from JSON
        if 'measures_to_calculate' in config_data and isinstance(config_data['measures_to_calculate'], str):
            config_data['measures_to_calculate'] = config_data['measures_to_calculate'].split(',')

        # Filter out any keys not in Config dataclass
        valid_keys = {f.name for f in fields(Config)}
        filtered_config_data = {k: v for k, v in config_data.items() if k in valid_keys}

        return Config(**filtered_config_data)

