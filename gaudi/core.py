import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import csr_matrix, issparse
from torch_geometric.nn import GCNConv

from .tools.config import DetectionConfig, GaudiArgumentParser
from .data.gaudi_data import GaudiData
from .models.representations import RepresentationsGenerator
from .models.soft_programs import SoftProgramsGenerator
from .tools.analysis import generate_labels as ext_generate_labels, compare as ext_compare
from .tools.measures import (
    compute_basic_interactions,
    compute_basic_measure,
    compute_higher_order_interactions,
    compute_higher_order_programs_scores,
)
from .tools.measures import compute_compositions as ext_compute_compositions
from .tools.visualizations import (
    plot_dotplot as ext_plot_dotplot,
    plot_composition as ext_plot_composition,
    plot_communities as ext_plot_communities,
    plot_dots as ext_plot_dots
)

EPS = 1e-10


class GaudiObject:
    """
    Manages and processes spatially resolved transcriptomics data through comprehensive data integration and analysis functionalities,
    based on Scanpy's facilities for individual-level and community-level biological insights.

    Attributes:
        raw_adata (Union[List[ad.AnnData], ad.AnnData]): Raw experimental data.
        sample_name (str): Identifier for the sample.
        dataset_name (str): Name of the dataset.
        experiment_name (str): Identifier for the experiment.
        pca_components (int): Number of principal components used in PCA.
        min_counts (int): Minimum counts per cell required for filtering.
        max_counts (int): Maximum counts per cell for filtering; None means no upper limit.
        min_genes (int): Minimum number of genes expressed per cell.
        min_cells (int): Minimum number of cells in which a gene must be expressed.
        max_pct_mt (int): Maximum allowed percentage of mitochondrial gene counts.
        n_top_genes (int): Number of top highly variable genes to keep.
        cellchatdb_type (str): Specifies the type of CellChat database, such as 'Human' or 'Mouse'.
        max_edges_length (int): Maximum spatial distance between cells to consider them connected.
        k_nearest_neighbors (int): Number of nearest neighbors considered in neighborhood graphs.
        community_detection_method (str): Method used for detecting cell communities.
        first_step_detection_config (DetectionConfig): Configuration for the initial step of community detection.
        rest_steps_detection_config (DetectionConfig): Configuration for subsequent steps of community detection.
        n_iter (int): Number of iterations for the analysis.
        measures_to_calculate (List[str]): List of measures to calculate during analysis.
        main_composition_key (Optional[str]): Key for the main composition.
        seed (int): Seed for random number generation to ensure reproducibility.
        config (GaudiArgumentParser): Configuration object parsed from input arguments.
    """

    def __init__(
        self,
        raw_adata: Union[List[ad.AnnData], ad.AnnData] = None,
        sample_name: str = "default_sample",
        dataset_name: str = "default_dataset",
        experiment_name: str = "default_experiment",
        pca_components: int = 100,
        min_counts: int = 10,
        max_counts: int = None,
        min_genes: int = 10,
        min_cells: int = 30,
        max_pct_mt: int = 20,
        n_top_genes: Optional[int] = None,
        cellchatdb_type: str = "Human",
        max_edges_length: int = None,
        k_nearest_neighbors: int = 15,
        community_detection_method: str = "optics",
        first_step_detection_config: DetectionConfig = None,
        rest_steps_detection_config: DetectionConfig = None,
        n_iter: int = 1,
        measures_to_calculate: List[str] = [
            "mean",
            "max_ratio",
            "var",
            "total_var",
            "lr_pathways",
            "programs",
        ],
        main_composition_key: Optional[str] = None,
        seed: int = 1,
        config: GaudiArgumentParser = None,
    ):
        self.raw_adata = raw_adata
        self.sample_name = sample_name
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.cellchatdb_type = cellchatdb_type
        self.pca_components = pca_components
        self.min_counts = min_counts
        self.max_counts = max_counts
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.max_pct_mt = max_pct_mt
        self.n_top_genes = n_top_genes
        self.max_edges_length = max_edges_length
        self.k_nearest_neighbors = k_nearest_neighbors
        self.first_step_detection_config = first_step_detection_config
        self.rest_steps_detection_config = rest_steps_detection_config
        self.community_detection_method = community_detection_method
        self.measures_to_calculate = measures_to_calculate
        self.main_composition_key = main_composition_key
        
        self.n_iter = n_iter
        self.seed = seed

        self.config = self.update_by_config(config)
        self.preprocessing_config = self._set_preprocessing_config()

        self.cellchatdb_path = self._get_cellchatdb_path()

        self.experiment_path = self._create_experiment_folders()
        self.geneset_db_path = self._get_geneset_db_path()
        self._set_seed()
        self.multilevel_spatial_graph = self._create_or_load_data()

        self.sample = self._init_order_perspectivess()
        self._set_main_composition_key()
        self.compute_measures(self.measures_to_calculate)
        
        self._init_log1p_base()
        self.save_config()

    def _set_preprocessing_config(self):
        """
        Sets the adata preprocessing configuration.
        """
        return {
            "pca_components": self.pca_components,
            "min_counts": self.min_counts,
            "max_counts": self.max_counts,
            "min_cells": self.min_cells,
            "min_genes": self.min_genes,
            "max_pct_mt": self.max_pct_mt,
            "n_top_genes": self.n_top_genes,
        }
    
    def _get_community_detection_config(self, config_params: dict = None):
        """
        Returns an updated configuration for the community detection stage.

        Returns:
            dict: community detection configuration.
        """
        config = {
            "initial_method": "optics",
            "initial_min_samples": 5,
            "initial_min_cluster_size": 10,
            "initial_xi": 0.01,
            "initial_max_eps": np.inf,
            "refinement_method": "optics",
            "refinement_min_samples": 5,
            "refinement_xi": 0.01,
            "refinement_max_eps": np.inf,
            "refinement_min_cluster_size": 5,
            "refinement_n_iter": 1,
        }

        if config_params:
            config.update(config_params)
        return config

    def _set_main_composition_key(self):
        """
        Sets the main composition key.
        """
        if self.main_composition_key is not None:
            if not isinstance(self.main_composition_key, str):
                raise ValueError("main_composition_key must be a string.")
            if self.main_composition_key not in self.raw_adata.obs.columns:
                raise ValueError(
                    f"main_composition_key {self.main_composition_key} not found in the raw anndata obs."
                )
            self.sample.labels[0][self.main_composition_key] = self.sample.base_adata.obs[self.main_composition_key].values
        
    def _create_experiment_folders(self):
        """Creates folders for storing experiment outputs."""
        base_path = Path("./gaudi_experiments")
        self.experiments_folder_path = base_path / self.dataset_name
        self.experiments_folder_path.mkdir(parents=True, exist_ok=True)
        experiment_path = self.experiments_folder_path / self.experiment_name
        self.checkpoints_path = experiment_path / "checkpoints"
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        self.figures_path = experiment_path / "figures"
        self.figures_path.mkdir(parents=True, exist_ok=True)
        return experiment_path

    def _get_geneset_db_path(self):
        return (
            self.experiment_path
            / "processed"
            / self.sample_name
            / f"{self.sample_name}_geneset_db.csv"
        )

    def _set_seed(self):
        """Sets the seed for random number generation."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # for multi-GPU.

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _create_or_load_data(self):
        """Initializes or loads the GaudiData object."""
        return GaudiData(
            raw_adata=self.raw_adata,
            dataset_name=self.dataset_name,
            sample_name=self.sample_name,
            gaudi_experiments_path=self.experiment_path,
            max_edges_length=self.max_edges_length,
            k_nearest_neighbors=self.k_nearest_neighbors,
            community_detection_method=self.community_detection_method,
            first_step_detection_config=self.first_step_detection_config,
            rest_steps_detection_config=self.rest_steps_detection_config,
            n_iter=self.n_iter,
            preprocessing_config=self.preprocessing_config,
        )

    def _init_order_perspectivess(self):
        """
        Initializes HigherOrderData for each sample in the dataset.

        This method creates an instance of HigherOrderData for each given sample name,
          preparing the higher-order structure required for further processing.

        Returns:
            HigherOrderData object.
        """

        sample_data = self.multilevel_spatial_graph[0]
        processed_dir_path = self.experiment_path / "processed" / self.sample_name

        higher_order_data = HigherOrderData(
            sample_name=self.sample_name,
            processed_dir_path=processed_dir_path,
            separators=sample_data.separators,
            primitives=sample_data.primitives,
            primitive_to_idx=sample_data.primitive_to_idx,
            idx_to_primitive=sample_data.idx_to_primitive,
            pos=sample_data.pos,
            communities_coordinates=sample_data.communities_coordinates,
        )

        sample = higher_order_data
        community_list = sample.primitives[sample.separators[1] : sample.separators[2]]
        community_sizes = [len(fs) for fs in community_list]
        average_community_size = sum(community_sizes) / len(community_sizes)
        median_community_size = np.median(community_sizes)
        sample.avg_num_cells_per_community = average_community_size
        sample.median_num_cells_per_community = median_community_size
        sample.avg_degree = sample_data.avg_degree

        return sample
    
    @property
    def perspectives(self):
        """
        Returns the perspectives of the sample.
        """
        return self.sample.perspectives

    @property
    def labels(self):
        """
        Returns the labels of the sample.
        """
        return self.sample.labels
        
    def _add_community_assignments(self, adata):
        communities = self.sample.primitives[self.sample.separators[1] :]
        adata.obs["community"] = -1
        for community_index, community in enumerate(communities):
            # Update the 'community' column for indices in this community
            for individual in community:
                adata.obs.iloc[individual, adata.obs.columns.get_loc("community")] = (
                    community_index
                )
        return adata

    def compute_measures(self, measures_to_calculate=[]):
        """
        Computes specified types of aggregated data.

        Args:
            measures_to_calculate (list): List of aggregation types to calculate.
        """

        if type(measures_to_calculate) != list:
            raise ValueError("measures_to_calculate must be a list.")

        if "mean" not in measures_to_calculate:
            measures_to_calculate.append("mean")

        aggregation_settings = {
            "mean": [0, 1],
            "max_ratio": [1],
            "var": [1],
            "total_var": [1],
        }

        for aggregation_type, levels in aggregation_settings.items():
            if aggregation_type in measures_to_calculate:
                for level in levels:
                    adata_path = (
                        self.experiment_path
                        / "processed"
                        / self.sample_name
                        / f"{self.sample_name}_{aggregation_type}_level_{level}.h5ad"
                    )

                    if self.load_if_exists(adata_path, aggregation_type, level):
                        continue

                    if level == 0:
                        adata = compute_basic_measure(
                            self.sample, aggregation_type=aggregation_type, level=level
                        )
                    else:
                        adata = compute_basic_measure(
                            self.sample, aggregation_type=aggregation_type, level=level, main_composition_key=self.main_composition_key
                        )

                    if level == 0:
                        adata = self._add_community_assignments(adata)

                    self.sample.perspectives[f"{aggregation_type}"][level] = (
                        self._process_and_save_data(adata, level, adata_path)
                    )

        if "lr_pathways" in measures_to_calculate:
            self.compute_lr_based_pathways_scores()

        if "programs" in measures_to_calculate:
            self.generate_programs()
            self.compute_programs_scores()

    def _get_cellchatdb_path(self):
        """
        Returns the path to the CellChat database file based on the cellchatdb_type.

        Args:
            cellchatdb_type (str): The type of CellChat database.

        Returns:
            Path: The path to the CellChat database file.
        """
        if self.cellchatdb_type not in ["Human", "Mouse"]:
            raise ValueError(
                f"Invalid cellchatdb_type. Must be one of ['Human', 'Mouse']."
            )

        return Path(f"assets/cellchatdb/{self.cellchatdb_type}.csv")

    def load_if_exists(self, path, key, level):
        """
        Loads an AnnData object if it exists.

        Args:
            path (str): The path to the AnnData object.
            key (str): The key to the object in the sample's perspectives.
            level (int): The level of the object.

        Returns:
            ad.AnnData: The loaded AnnData object.
        """

        if os.path.exists(path):

            logging.info(
                f"Found an existing Level {level} {key}-based object. Loading from {path}..."
            )
            self.sample.perspectives[key][level] = ad.read_h5ad(path)
            return True
        return False

    def compute_lr_based_pathways_scores(self):

        for level in [0, 1]:
            lr_pathways_path = (
                self.experiment_path
                / "processed"
                / self.sample_name
                / f"{self.sample_name}_lr_pathways_{level}.h5ad"
            )

            if self.load_if_exists(lr_pathways_path, "lr_pathways", level):
                continue

            if level == 0:
                if self.cellchatdb_path is None:
                    logging.warning(
                        "No LR database provided. Cannot calculate LR-based pathway scores."
                    )
                    continue

            if level == 0:
                adata = compute_basic_interactions(self.sample, self.cellchatdb_path)
            else:
                adata = compute_higher_order_interactions(
                    self.sample, self.multilevel_spatial_graph[0]
                )

            if adata is None:
                logging.warning(
                    f"No LR-based pathways scores were generated for level {level}."
                )
                return

            self.sample.perspectives["lr_pathways"][level] = (
                self._process_and_save_data(adata, level, lr_pathways_path)
            )

    def generate_representations(
        self,
        train=True,
        hidden_dim=1000,
        output_dim=100,
        device=None,
        lr=1e-4,
        weight_decay=1e-5,
        gamma=0.1,
        patience=10,
        epochs=500,
        conv_layer=GCNConv,
        num_node_layers=2,
        num_community_layers=2,
        seed=1,
        **kwargs,
    ):
        """
        Generates cell and community level representations using the Heirarchical Deep Graph Infomax learning procedure integrated with the Community-Focused Networks architecture.
        """
        representations_generator = RepresentationsGenerator(
            sample=self.sample,
            data=self.multilevel_spatial_graph[0],
            checkpoints_path=self.checkpoints_path,
        )

        representations_generator.train_or_load_model(
            train=train,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            gamma=gamma,
            patience=patience,
            epochs=epochs,
            conv_layer=conv_layer,
            activation_fn=activation_fn,
            num_layers=num_layers,
            seed=seed,
            **kwargs,
        )
        embeds = representations_generator.get_embedding_list()
        self._update_embedding(embeds)

    def _update_embedding(self, embedding):
        for level in range(2):
            self.sample.embeds[level] = embedding[level]
            for key, adatas in self.sample.perspectives.items():
                if key == "lr_pathways" and level == 0:
                    continue
                if adatas[level] is not None:
                    adatas[level].obsm["gaudi_emb"] = embedding[level]

    def _project_community_labels_to_cells(self, level, labels_key):
        """
        Projects the community labels to the cells in the sample.

        Args:
            level (int): The order level.
            labels_key (str): The name of the labels.
        """

        community_labels = self.sample.labels[level][labels_key]
        primitives = self.sample.primitives
        separators = self.sample.separators
        communities = primitives[separators[1] : separators[2]]

        cell_ids = set(range(separators[0], separators[1]))
        projected_labels = np.empty(separators[1], dtype="U50")

        for community_idx, community in enumerate(communities):
            community_label = community_labels[community_idx]
            for cell_idx in community:
                projected_labels[cell_idx] = community_label
                cell_ids.remove(cell_idx)

        for cell_idx in cell_ids:
            projected_labels[cell_idx] = "999999999"

        if self.sample.labels[0] is None:
            self.sample.labels[0] = {}

        projected_labels = pd.Series(projected_labels).astype("str").values

        projected_labels_key = labels_key + "_Projected_from_Community"
        self.sample.labels[0][projected_labels_key] = projected_labels

        return projected_labels_key

    def _update_labels(self, level):
        """
        Updates the labels in the sample's labels list at the specified order level.

        Args:
            level (int): The order level.
            labels (np.array): The labels to be added.
        """

        for labels_key in self.sample.labels[level]:

            labels = self.sample.labels[level][labels_key]

            if level == 0:
                self.sample.base_adata.obs[labels_key] = labels
                self.sample.base_adata.obs[labels_key] = self.sample.base_adata.obs[
                    labels_key
                ].astype("category")

            for key, adatas in self.sample.perspectives.items():
                if level == 0:
                    if key == "lr_pathways":
                        continue

                if adatas[level] is not None:
                    adatas[level].obs[labels_key] = labels
                    adatas[level].obs[labels_key] = (
                        adatas[level].obs[labels_key].astype("category")
                    )

            if level == 1:
                projected_labels_key = self._project_community_labels_to_cells(
                    level, labels_key
                )
                self._update_labels(0)

    def generate_labels(
        self,
        embedding_key: Optional[str] = None,
        level: int = 1,
        clustering_method: str = "leiden",
        layer: str = None,
        labels_name: str = "gaudi_labels",
        n_clusters: int = 10,
        resolution: float = 0.5,
        neighbors_metric: str = "cosine",
        n_neighbors: int = 100,
    ):

        ext_generate_labels(
            self.sample,
            embedding_key,
            level,
            clustering_method,
            layer,
            labels_name,
            n_clusters,
            resolution,
            neighbors_metric,
            n_neighbors,
        )

        self._update_labels(level)

    def generate_programs(
        self,
        data_type="mean",
        layer="counts_based",
        encoding_dim=50,
        n_programs=250,
        with_pca=True,
        pca_components=425,
        probability_threshold=0.95,
        epochs=500,
        batch_size=400,
        layer_multipliers=[1, 2, 0.75, 0.5, 0.25],
        device="cpu",
        lr=0.001,
        factor=0.1,
        patience=10,
        seed=1,
    ):

        if os.path.exists(self.geneset_db_path):
            logging.info(
                f"Found an existing programs' gene set database file at {self.geneset_db_path}."
            )
            return
        else:
            logging.info(
                f"Detecting programs and storing their gene sets in {self.geneset_db_path}."
            )

        if (
            data_type not in self.sample.perspectives
            or self.sample.perspectives[data_type][1] is None
        ):
            raise ValueError(
                f"Data type {data_type} not found in level 1 of the sample perspectives."
            )
        adata = self.sample.perspectives[data_type][1]
        programs_generator = SoftProgramsGenerator(
            adata,
            self.geneset_db_path,
            layer=layer,
            encoding_dim=encoding_dim,
            n_clusters=n_programs,
            with_pca=with_pca,
            pca_components=pca_components,
            probability_threshold=probability_threshold,
            epochs=epochs,
            batch_size=batch_size,
            layer_multipliers=layer_multipliers,
            device=device,
            lr=lr,
            factor=factor,
            patience=patience,
            seed=seed,
        )
        programs_generator.run_pipeline()

    def compute_programs_scores(self):

        for level in [0, 1]:
            programs_path = (
                self.experiment_path
                / "processed"
                / self.sample_name
                / f"{self.sample_name}_programs_{level}.h5ad"
            )

            if self.load_if_exists(programs_path, "programs", level):
                continue

            adata = compute_higher_order_programs_scores(
                self.sample, self.geneset_db_path, level
            )

            self.sample.perspectives["programs"][level] = self._process_and_save_data(
                adata, level, programs_path
            )

    def _process_and_save_data(self, adata, level, save_path, n_comps=50):
        """
        Processes and saves the AnnData object, following standard preprocessing steps.
        """

        pos = self.multilevel_spatial_graph[0].pos
        separators = self.sample.separators
        ids = np.array(range(separators[level], separators[level + 1]))

        if pos[ids].shape[0] == adata.shape[0]:
            adata.obsm["spatial"] = pos[ids]
            adata.obs["id"] = ids

        if not issparse(adata.X):
            adata.X = csr_matrix(adata.X)
            
        if not (adata.X.todense() < 0).any():
            sc.pp.normalize_total(adata, inplace=True, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.uns["log1p"]["base"] = None
            adata.layers["log_normalized_counts"] = csr_matrix(adata.X.copy())

            if n_comps < adata.shape[1] and n_comps < adata.shape[0]:
                sc.pp.scale(adata, zero_center=False)
                try:
                    sc.pp.pca(adata, n_comps=n_comps)
                except Exception as e:
                    print(f"{str(e)}")
                    pass

        adata.X = adata.layers["log_normalized_based"].copy()

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            adata.write(save_path)
            return adata
        except Exception as e:
            print(f"Error while saving object to {save_path}: {str(e)}")
            return None

    def plot_dots(
        self,
        level: int,
        color_by: List[str],
        data_type: str = None,
        layer: str = None,
        size: int = None,
        figsize: Tuple[int, int] = None,
        title: str = None,
        dpi: int = None,
        save: Optional[Union[bool, str]] = True,
        format: str = "png",
        palette: str = None,
        cmap: str = "viridis",
        invert_yaxis: bool = False,
        invert_xaxis: bool = False,
        **kwargs: Any,
    ):
        """
        Visualizes the spatial position of cells or communities depending on the level specified.
        Level 0 visualizes individual cells; level 1 visualizes communities as mean locations of cells.

        Args:
            level (int): Level of data visualization (0 for cells, 1 for communities).
            color_by (List[str]): Features or metadata to color the dots by.
            data_type (str, optional): Type of data to fetch ('mean', 'median', etc.), affects how communities are aggregated.
            layer (str, optional): Specific layer of data to use.
            size (int, optional): Size of the dots.
            figsize (Tuple[int, int], optional): Size of the figure.
            title (str, optional): Title of the plot.
            dpi (int, optional): Dots per inch for the plot resolution.
            save (Union[bool, str], optional): Path or flag to save the plot.
            format (str): Format to save the plot in.
            palette (str, optional): Color palette to use.
            cmap (str): Colormap for continuous features.
            invert_yaxis (bool): Whether to invert the y-axis.
            invert_xaxis (bool): Whether to invert the x-axis.
        """
        # Access the appropriate data layer or base AnnData depending on data_type
        adata = self.sample.base_adata if data_type is None else self.sample.perspectives[data_type][level]

        # Determine the save path if needed
        if isinstance(save, str):
            save_path = Path(save)
        elif save:
            filename = f"dots_{self.sample_name}_{data_type}_{color_by}_{level}.{format}"
            save_path = self.figures_path / filename
        else:
            save_path = None

        # Set up the title for the plot, defaulting to the combined parameters if not provided
        if not title:
            title = f"{color_by} - {self.sample_name}"

        # Call to the external plot function with the appropriate parameters
        ext_plot_dots(
            adata=adata,
            color=color_by,
            layer=layer,
            size=size,
            figsize=figsize,
            title=title,
            dpi=dpi,
            save_path=save_path,
            palette=palette,
            cmap=cmap,
            invert_yaxis=invert_yaxis,
            invert_xaxis=invert_xaxis,
            **kwargs,
        )

    def plot_communities(
        self,
        color_by: Optional[Union[str, List[str]]] = None,
        data_type: str = "mean",
        with_points: bool = False,
        layer: str = "X",
        figsize: tuple = (8, 6),
        title: Optional[str] = None,
        dpi: int = 100,
        save: Optional[Union[bool, str]] = None,
        format: str = "png",
        background_color: str = "white",
        vmax: Optional[float] = None,
        custom_label_names: Optional[List[str]] = None,
        invert_yaxis: bool = False,
        invert_xaxis: bool = False,
        **kwargs,
    ) -> None:
        """Plots communities, colored by the given attribute (color_by).

        Args:
            color_by: The attribute or metadata to color communities by.
            data_type: Type of data to fetch ('mean', 'var', etc.).
            with_points: Whether to include cell level positions in the plot.
            layer: Specify the layer of data to use from AnnData object in case the attribute is continues.
            figsize: Size of the figure.
            title: Title of the plot.
            dpi: Dots per inch (resolution) of the figure.
            save: Path or boolean to save the figure.
            format: Format to save the figure in.
            background_color: Background color of the plot.
            vmax: Maximum data value for normalization.
            custom_label_names: Custom names for the labels.
            invert_yaxis: Invert the y-axis.
            invert_xaxis: Invert the x-axis.
        """
        sample = self.sample
        sample_name = self.sample_name
        points = sample.base_adata.obsm["spatial"] if with_points else None
        communities_coordinates = sample.communities_coordinates
        adata = sample.perspectives[data_type][1]

        # Determine save path
        save_path = save if isinstance(save, str) else self.figures_path / f"communities_{sample_name}_{color_by}_{data_type}"
        os.makedirs(self.figures_path, exist_ok=True)

        # Set the plot title
        plot_title = f"{color_by} - {sample_name}" if title is None else title

        # Validate and process color_by parameter
        if isinstance(color_by, str):
            if color_by in sample.labels[1]:
                labels = sample.labels[1][color_by]
            elif color_by in adata.var_names:
                labels = adata[:, color_by].X if layer == "X" else adata[:, color_by].layers[layer]
            else:
                raise ValueError(f"{color_by} is neither a recognized feature nor a label.")

            # Call to external plot function
            ext_plot_communities(
                communities_coordinates=communities_coordinates,
                points=points,
                labels=labels,
                title=plot_title,
                figsize=figsize,
                save_path=save_path,
                file_format=format,
                dpi=dpi,
                background_color=background_color,
                vmax=vmax,
                custom_label_names=custom_label_names,
                invert_yaxis=invert_yaxis,
                invert_xaxis=invert_xaxis,
                **kwargs,
            )
        else:
            raise ValueError("color_by must be a string.")

    def update_by_config(self, config: GaudiArgumentParser):
        """
        Updates the GaudiObject with the provided configuration.
        """
        if config:
            self.config = config
            for attr in self.__dict__:
                if hasattr(config, attr) and getattr(config, attr) is not None:
                    setattr(self, attr, getattr(config, attr))

        return config

    def save_config(self):
        """
        Saves the configuration to the experiment folder.
        """
        # save the configuration to the experiment folder
        config_path = self.experiment_path / "config.json"

        if config_path.exists():
            return
        config_as_dict = self.config.__dict__
        if isinstance(config_as_dict["first_step_detection_config"], DetectionConfig):
            config_as_dict["first_step_detection_config"] = asdict(
                config_as_dict["first_step_detection_config"]
            )
            config_as_dict["rest_steps_detection_config"] = asdict(
                config_as_dict["rest_steps_detection_config"]
            )

        with open(config_path, "w") as f:
            json.dump(config_as_dict, f, indent=4)

    def compute_compositions(
        self,
        group_1,
        group_2,
        group_1_level,
        group_2_level,
        group_1_level_prefix=None,
        group_2_level_prefix=None,
    ):
        """
        Compute the composition of group_2 categories per group_1 label and per community (tile).

        Args:
            group_1 (str): The group 1's labels name.
            group_2 (str): The group 2's labels name.
            group_1_level (int): The group 1 level.
            group_2_level (int): The group 2 level.
            group_1_level_prefix (str, optional): The group 1 level prefix. Defaults to None.
            group_2_level_prefix (str, optional): The group 2 level prefix. Defaults to None.

        Returns:
            tuple: A tuple containing two DataFrames:
                - Composition of group_2 categories per group_1 label.
                - Composition of group_2 categories per community (tile).
        """
        sample = self.sample
        group_2_composition_per_group_1, group_2_distribution_per_community_df = (
            ext_compute_compositions(
                sample,
                group_1,
                group_2,
                group_1_level,
                group_2_level,
                group_1_level_prefix=group_1_level_prefix,
                group_2_level_prefix=group_2_level_prefix,
            )
        )

        self.perspectives["composition"][
            group_1_level
        ] = group_2_distribution_per_community_df

        self._update_labels(group_1_level)

        return group_2_composition_per_group_1, group_2_distribution_per_community_df

    def plot_composition(
        self,
        group_1: str,
        group_2: str,
        group_1_level: int,
        group_2_level: int,
        group_1_level_prefix: Optional[str] = None,
        group_2_level_prefix: Optional[str] = None,
        save: Optional[bool] = True,
        custom_group_1_labels: Optional[Dict[int, str]] = None,
        custom_group_2_labels: Optional[Dict[int, str]] = None,
        **kwargs,
    ):
        """
        Plot the composition of group_2 categories per group_1 label.

        Args:
            group_1 (str): The group 1's labels name.
            group_2 (str): The group 2's labels name.
            group_1_level (int): The group 1 level.
            group_2_level (int): The group 2 level.
            group_1_level_prefix (str, optional): The group 1 level prefix. Defaults to None.
            group_2_level_prefix (str, optional): The group 2 level prefix. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to None.
            custom_group_1_labels (Dict[int, str], optional): Custom labels for group 1. Defaults to None.
            custom_group_2_labels (Dict[int, str], optional): Custom labels for group 2. Defaults to None.
            **kwargs: Additional keyword arguments for plotting.
        """
        sample_name = self.sample_name

        # Create plot directory if it does not exist
        os.makedirs(self.figures_path, exist_ok=True)
        save_path = (
            self.figures_path / f"{sample_name}_{group_1}_{group_2}_composition"
            if save
            else None
        )

        # Compute compositions
        group_2_composition_per_group_1, group_2_distribution_per_community_df = (
            self.compute_compositions(
                group_1,
                group_2,
                group_1_level,
                group_2_level,
                group_1_level_prefix=group_1_level_prefix,
                group_2_level_prefix=group_2_level_prefix,
            )
        )

        # Transpose for plotting
        group_2_composition_per_group_1 = group_2_composition_per_group_1.transpose()

        # Plot composition
        ext_plot_composition(
            group_2_composition_per_group_1,
            group_1,
            group_2,
            save_path,
            custom_group_1_labels,
            custom_group_2_labels,
            **kwargs,
        )

    def plot_dotplot(
        self,
        data_type: str,
        level: int,
        var_names: Dict[str, List[str]],
        groupby: str,
        groups: Union[str, List[str]] = "all",
        figsize: tuple = (25, 5),
        layer: str = "log_normalized_counts",
        standard_scale: str = "var",
        min_counts: int = 1,
        exclude_labels: Optional[List[str]] = None,
        dendrogram: bool = True,
        grid: bool = True,
        cmap: str = "Reds",
        largest_dot: int = 200,
        legend_width: int = 3,
        save: Optional[Union[bool, str]] = True,
        **kwargs
    ) -> None:
        """
        Plots a dotplot from an AnnData object located within a nested structure of the sample.
        Automatically manages paths for saving figures based on internal and provided identifiers.
        
        Parameters:
            - data_type: Type of data the plot is based on.
            - level: Data hierarchy level.
            - var_names: Variable names to plot, specified per group (class).
            - groupby: Labels group name to group by.
            - groups: Specific groups to include or 'all'.
            - figsize: Size of the figure.
            - layer: Data layer the plot is based on.
            - save: Path or flag to save the figure.
            - Other plotting parameters.
        
        Returns:
            None
        """
        sample = self.sample
        sample_name = self.sample_name
        adata = sample.perspectives[data_type][level].copy()

        if isinstance(save, str):
            save_path = Path(save)
        elif save is True:
            save_path = self.figures_path / f"dotplot_{sample_name}_{groupby}_vs_{groups}_{data_type}_{layer}.png"
        else:
            save_path = None

        if save_path:
            os.makedirs(self.figures_path, exist_ok=True)

        ext_plot_dotplot(
            adata,
            var_names=var_names,
            groupby=groupby,
            groups=groups,
            figsize=figsize,
            standard_scale=standard_scale,
            min_counts=min_counts,
            exclude_labels=exclude_labels,
            dendrogram=dendrogram,
            grid=grid,
            cmap=cmap,
            largest_dot=largest_dot,
            legend_width=legend_width,
            save_path=save_path,
            **kwargs,
        )

    def compare(
        self,
        data_type: str,
        level: int,
        groupby: str,
        groups: Any = "all",
        reference: str = "rest",
        method: str = "wilcoxon",
        layer: Any = None,
        min_in_group_fraction: float = 0.25,
        min_fold_change: float = 0.2,
        max_out_group_fraction: float = 1.1,
        compare_abs: bool = True,
        tie_correct: bool = True,
        n_features: int = 10,
        key: str = "ranked_features_groups",
        diff_pct: Any = None,
        min_counts: int = 1,
        exclude_labels: Any = None,
        max_pvalue: float = 0.05,
        **kwargs,
    ) -> Tuple[List[Any], Dict[str, List[Any]]]:
        """
        Performs a statistical comparison between specified groups within the dataset using Scanpy's ranking and filtering functions. This method is a wrapper around scanpy.tl.rank_genes_groups and scanpy.tl.filter_rank_genes_groups, tailored to facilitate easy access to differential expression analysis within a structured data hierarchy.

        Args:
            data_type (str): Specifies the type of data to perform the comparison on.
            level (int): Indicates the level within the perspectives hierarchy from which data is drawn.
            groupby (str): The key in the observation DataFrame used to define groups.
            groups (Union[str, List[str], Set[str]], optional): Defines specific groups to include for comparison, or "all" to include all available groups.
            reference (str, optional): Specifies the reference group against which other groups are compared.
            method (str, optional): Statistical method used for comparison (e.g., 'wilcoxon', 't-test').
            layer (str, optional): Specifies the layer of data within the AnnData object to analyze.
            min_in_group_fraction (float, optional): Minimum fraction of data points required within each group for it to be included in the analysis.
            min_fold_change (float, optional): Minimum fold change required to consider a gene significantly differentially expressed.
            max_out_group_fraction (float, optional): Maximum allowable fraction of data points in out-groups for inclusion in the analysis.
            compare_abs (bool, optional): If True, compares the absolute values instead of the actual values.
            tie_correct (bool, optional): If True, applies tie correction in the statistical testing process.
            n_features (int, optional): Number of top features (genes) to return after filtering.
            key (str, optional): Key under which to store the results within the AnnData object's `.uns` attribute.
            diff_pct (float, optional): Differential percentage threshold for filtering significant features.
            min_counts (int, optional): Minimum count of data points required to include a label in the analysis.
            exclude_labels (List[str], optional): Specific labels to exclude from the analysis.
            max_pvalue (float, optional): Maximum adjusted p-value to consider features statistically significant.
            **kwargs: Additional keyword arguments passed to Scanpy's rank_genes_groups and filter_rank_genes_groups functions.

        Returns:
            Tuple[List[Any], Dict[str, List[Any]]]: Non-NaN values and a dictionary containing subsets of significant features organized by group.
        """

        adata = self.sample.perspectives[data_type][level].copy()
        non_nan_values, subsets_dict = ext_compare(
            adata,
            groupby,
            groups,
            reference,
            method,
            layer,
            min_in_group_fraction,
            min_fold_change,
            max_out_group_fraction,
            compare_abs,
            tie_correct,
            n_features,
            key,
            diff_pct,
            min_counts,
            exclude_labels,
            max_pvalue,
            **kwargs,
        )
        return non_nan_values, subsets_dict
    
    def __str__(self):
        return f"GaudiObject for sample '{self.sample_name}' from the '{self.dataset_name}' dataset, experiment '{self.experiment_name}'."

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        # In the future this could be the number of samples in the dataset
        return len(self.raw_adata)
    

    def _init_log1p_base(self):
        for adata_list in list(self.sample.perspectives.values()):
            for adata in adata_list:
                if adata is not None:
                    if "log1p" in adata.uns:
                        adata.uns["log1p"]["base"] = None


@dataclass
class HigherOrderData:
    """
    Serves as a container for managing data across different scales of biological organization.

    This class maintains multiple objects of varying types, categorized into two levels:
    - level 0:  representing individual-level data.
    - level 1:  representing community-level data.

    Attributes:
        sample_name (str): Identifier for the sample.
        processed_dir_path (Path): Path to where the sample's processed data is stored.
        separators (List[int]): Index markers that separate different hierarchies within the primitives.
        primitives (List[frozenset]): Sets representing the entities. A cell is represented as a set of one index,
                                      while a community is represented as a set of multiple indices.
        primitive_to_idx (Dict[frozenset, int]): Maps entities to their indices for quick access.
        idx_to_primitive (Dict[int, frozenset]): Reverse map from indices back to entities.
        pos (np.ndarray): Array of spatial positions for the entities.
        communities_coordinates (List[np.ndarray]): Spatial coordinates that define the spatial structure of each community.
        embeds (List[Optional[np.array]]): Embeddings for cells and communities, generated by Gaudi.
        perspectives (Dict[str, List[Optional[ad.AnnData]]]): Structured data views for different analysis angles, such as expression levels or interaction metrics.
        labels (Dict[int, Dict[str, Union[str, int]]]): Categorical labels for data at different hierarchical levels.
        label_counts (Dict[int, int]): Counts of labels, providing summaries at each hierarchical level.
        base_adata (ad.AnnData): The base AnnData object, post basic preprocessing.
        avg_num_cells_per_community (Optional[float]): Average number of cells per community.
        median_num_cells_per_community (Optional[float]): Median number of cells per community.
        avg_degree (Optional[float]): Average connectivity degree among data elements.
    """

    sample_name: str
    processed_dir_path: Path
    separators: List[str]
    primitives: List[frozenset]
    primitive_to_idx: Dict[frozenset, int]
    idx_to_primitive: Dict[int, frozenset]
    pos: np.ndarray
    communities_coordinates: List[np.ndarray]
    embeds: List[np.array] = field(default_factory=lambda: [None, None])
    perspectives: Dict[str, List[ad.AnnData]] = field(
        default_factory=lambda: {
            key: [None, None]
            for key in [
                "mean",
                "max_ratio",
                "var",
                "total_var",
                "lr_pathways",
                "programs",
                "gnn",
                "composition",
            ]
        }
    )
    labels: Any = field(default_factory=lambda: [{}, {}])
    label_counts: Any = field(default_factory=lambda: [None, None])
    base_adata: ad.AnnData = field(init=False)
    avg_num_cells_per_community: float = field(init=False)
    median_num_cells_per_community: float = field(init=False)
    avg_degree: float = field(init=False)

    def __post_init__(self):
        self.base_adata = ad.read_h5ad(self.processed_dir_path / "base_adata.h5ad")
