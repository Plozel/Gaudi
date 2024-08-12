import logging
import sys
from pathlib import Path
from typing import List, Tuple

import anndata as ad
import networkx as nx
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csr_matrix, lil_matrix
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


from ..tools.utils import files_exist
from ..data.preprocessing import run_preprocessing
from .structures import CommunityComplex, create_graph_from_point_cloud
from ..tools.config import DetectionConfig


class GaudiData(Dataset):
    """A dataset class for processing and loading spatial transcriptomics data with the Gaudi framework.

    Inherits from PyTorch Geometric's Dataset class, adding functionalities for reading,
    preprocessing, and community detection on spatial transcriptomics data represented as
    AnnData object.

    Args:
        adata (ad.AnnData): AnnData object containing the sample information.
        dataset_name (str): The name of the dataset.
        sample_name (str): Name of the sample included in the dataset.
        max_edges_length (int): The threshold length for edges in the spatial expression graph.
        gaudi_experiments_path (Path): Base directory for the dataset's experiments.
        community_detection_config (dict): Parameters for the community detection algorithm.
        community_detection_with_refinement (bool): Whether to refine community detection results.
        transform (optional): Function to transform data objects.
        pre_transform (optional): Function to apply before saving data objects to disk.
        pre_filter (optional): Function to decide whether a data object should be included.

    Attributes:
        root (Path): Path to the dataset's base directory.
        community_complex (CommunityComplex): The community complex data structure for the current sample.
    """

    def __init__(
        self,
        raw_adata: ad.AnnData,
        dataset_name: str,
        sample_name: str,
        max_edges_length: int,
        k_nearest_neighbors: int,
        gaudi_experiments_path: Path,
        community_detection_method: str,
        first_step_detection_config: DetectionConfig,
        rest_steps_detection_config: DetectionConfig,
        n_iter: int,
        preprocessing_config: dict,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **kwargs,
    ):
        self.root = gaudi_experiments_path
        self.raw_adata = raw_adata
        self.sample_name = sample_name
        self.dataset_name = dataset_name
        self.k_nearest_neighbors = k_nearest_neighbors
        self.max_edges_length = max_edges_length
        self.community_detection_method = community_detection_method
        self.first_step_detection_config = first_step_detection_config
        self.rest_steps_detection_config = rest_steps_detection_config
        self.n_iter = n_iter
        self.data = None
        self.community_complex = None
        self.adata = None
        self.preprocessing_config = preprocessing_config
        self._save_base_adata(self.raw_adata, self.raw_dir)
        super().__init__(gaudi_experiments_path, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self) -> Path:
        """Path to the raw data directory."""
        return self.root / "raw"

    @property
    def processed_dir(self) -> Path:
        """Path to the processed data directory."""
        return self.root / "processed"

    def _save_base_adata(self, adata: ad.AnnData, dir_path: Path):
        """
        Saves AnnData object to the raw directory.
        """
        path = dir_path / self.sample_name / f"base_adata.h5ad"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            adata.write_h5ad(path)

    @property
    def raw_file_names(self) -> List[str]:
        raw_files = []

        files = list((self.raw_dir / self.sample_name).glob("*.h5ad"))
        if not files:
            raise ValueError(f"No raw files found for {self.sample_name}.")
        raw_files.extend(files)
        return raw_files

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.sample_name}/data.pt"]

    def _process(self):
        """Checks and processes raw data if processed data doesn't exist."""

        if files_exist(self.processed_paths):
            return

        if self.log and "pytest" not in sys.modules:
            print("Processing...", file=sys.stderr)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.process()

        if self.log and "pytest" not in sys.modules:
            print("Done!", file=sys.stderr)

    def process(self):
        print(
            f"\nNo multilevel spatial graph found for sample {self.sample_name}, processing initial data..."
        )

        self.initialize_data()
        self.compute_avg_pos()
        self.initialize_higher_representations_and_adjs()

        processed_data_path = self.processed_dir / self.sample_name / "data.pt"
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.data, processed_data_path)

        self.community_complex = None


    def read_base_adata(self):
        """Reads the base AnnData from the raw directory for the current sample.

        Returns:
            ad.AnnData: The AnnData object for the current sample.
        """

        current_preprocessed_adata_path = (
            self.processed_dir / self.sample_name / "base_adata.h5ad"
        )
        adata = sc.read_h5ad(current_preprocessed_adata_path)
        adata.var_names = [gene.upper() for gene in adata.var_names]

        return adata

    def initialize_data(self):
        """Initializes the community complex."""

        self.adata = run_preprocessing(adata=self.raw_adata, **self.preprocessing_config)
        point_cloud = self.adata.obsm["spatial"]
        graph_adjacency_matrix, edge_index = self.create_community_complex(point_cloud)

        # Update spatial connectivities in the AnnData object
        self.adata.obsp["spatial_connectivities"] = csr_matrix(graph_adjacency_matrix)
        self._save_base_adata(self.adata, self.processed_dir)

        features = self.get_init_features(self.adata)
        self.create_pytorch_geometric_data_object(features, edge_index, point_cloud)
        self.create_primitive_idx_dicts()

    def create_community_complex(self, point_cloud):
        """Generates a community complex from a point cloud."""

        se_graph = create_graph_from_point_cloud(
            point_cloud, self.max_edges_length, self.k_nearest_neighbors
        )
        logging.info(
            f"Graph average degree: {sum(deg for _, deg in se_graph.degree()) / len(se_graph):.2f}"
        )

        self.community_complex = CommunityComplex(
            points=point_cloud,
            local_graph=se_graph,
            detection_method=self.community_detection_method,
            first_step_detection_config = self.first_step_detection_config,
            rest_steps_detection_config = self.rest_steps_detection_config,
            n_iter=self.n_iter,
        )


        nodelist = sorted(se_graph.nodes())
        graph_adjacency_matrix = nx.to_scipy_sparse_array(se_graph, nodelist=nodelist)
        undirected_edges = [(edge[0], edge[1]) for edge in se_graph.edges()] + [
            (edge[1], edge[0]) for edge in se_graph.edges()
        ]
        edge_index = torch.tensor(undirected_edges).t().contiguous()

        return graph_adjacency_matrix, edge_index

    def get_init_features(self, adata):
        """Extracts initial feature set from AnnData based on PCA."""
        return torch.from_numpy(adata.obsm["X_pca"])

    def compute_avg_pos(self) -> np.ndarray:
        """
        Set the average positions of the primitives based on their point set.
        """

        self.data.pos = np.array(
            [
                np.mean(self.data.pos[list(primitive)], axis=0)
                for primitive in self.data.primitives
            ]
        )

    def initialize_higher_representations_and_adjs(self):
        """Prepares higher order features and adjacency matrices for the dataset."""

        # Extend node features for higher order representations
        input_dim = self.data.x.shape[1]
        extended_x = torch.zeros((self.data.n_primitives, input_dim))
        extended_x[: len(self.data.x), :] = self.data.x
        self.data.x = extended_x

        # Prepare adjacency matrix for community relationships
        community_adj = lil_matrix(
            (
                self.data.n_primitives,
                self.data.n_primitives,
            ),
            dtype=int,
        )

        logging.info(f"Number of edges: {len(self.community_complex.edges)}")
        logging.info("Calculating adjacencies...")
        for primitive in tqdm(self.data.primitives[self.data.separators[-2] :]):
            primitive_idx = self.data.primitive_to_idx[primitive]
            # Set feature for the primitive based on the mean of its constituents
            self.data.x[primitive_idx] = torch.mean(
                self.data.x[list(primitive)], axis=0
            )
            # Update adjacency for the primitive
            neighbored_nodes = [
                self.data.primitive_to_idx[frozenset([node])] for node in primitive
            ]
            community_adj[primitive_idx, neighbored_nodes] = 1

        # Convert to COO format for PyTorch compatibility
        community_adj = community_adj.tocoo()
        self.data.community_edge_index = torch.tensor(
            [community_adj.col, community_adj.row], dtype=torch.long
        )

        # Store coordinates of community centers
        self.data.communities_coordinates = (
            self.community_complex.communities_coordinates
        )

    def create_pytorch_geometric_data_object(self, x, edge_index, point_cloud):
        """Initializes the dataset's PyTorch Geometric Data object."""
        self.data = Data(x=x, edge_index=edge_index)
        self.data.pos = point_cloud
        self.data.num_nodes = self.data.x.shape[0]
        self.data.primitives = []
        self.data.communities_coordinates = []
        self.data.separators = self.get_separators()
        self.data.n_primitives = self.data.separators[-1]

        self.data.avg_degree, self.data.avg_num_of_cells, self.data.num_cells_per_community = self.get_community_complex_characteristics()

    def get_community_complex_characteristics(self):
        """Calculates the averages number of cells per community"""
        num_communities = len(self.community_complex.communities)
        cells_per_community = [len(community) for community in self.community_complex.communities]
        total_cells_in_communities = sum(cells_per_community)
        average_cells_per_community = total_cells_in_communities / num_communities if num_communities > 0 else 0
        degrees = [deg for node, deg in self.community_complex.local_graph.degree()]
        average_degree = sum(degrees) / len(self.community_complex.local_graph)

        logging.info(f"Average number of cells within each community: {average_cells_per_community:.2f}")

        return average_degree, average_cells_per_community, cells_per_community

    def get_separators(self):
        """Returns the separators for the primitives in the spatial structure."""

        # TODO: After removing the edge representations, the separators object is not needed anymore (can use the number of nodes and communities instead)
        n_nodes, n_edges, n_communities = self.community_complex.lengths
        separators = [0, n_nodes, n_nodes + n_communities]
        return separators

    def create_primitive_idx_dicts(self):
        self.data.primitive_to_idx = {}
        self.data.idx_to_primitive = {}

        # Assume primitives_list is sorted by length and then by lexicographic order
        primitives_list = self.community_complex.primitives
        self.data.primitives = primitives_list

        for primitive_idx, primitive in enumerate(primitives_list):
            self.data.primitive_to_idx[primitive] = primitive_idx
            self.data.idx_to_primitive[primitive_idx] = primitive

    def len(self):
        """Returns the number of processed files in the dataset."""
        return len(self.processed_file_names)

    def get(self, idx):
        """Loads and returns the data object at the specified index."""
        data = torch.load(self.processed_dir / self.processed_file_names[idx])
        return data
