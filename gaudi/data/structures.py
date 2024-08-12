import logging
import numpy as np
from typing import Dict, List, Union
from dataclasses import dataclass, asdict
from scipy.spatial import ConvexHull, Delaunay

from sklearn.cluster import OPTICS, DBSCAN, SpectralClustering
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from sklearn.cluster import HDBSCAN
from scipy.spatial import ConvexHull
import numpy as np
from sklearn.cluster import SpectralClustering
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import OPTICS, DBSCAN, SpectralClustering
import inspect
from ..tools.config import DetectionConfig
logging.basicConfig(level=logging.INFO)


class CommunityComplex:
    def __init__(
        self, points, local_graph, detection_method, first_step_detection_config, rest_steps_detection_config, n_iter, **kwargs
    ):

        self.local_graph = local_graph
        self.communities = None
        self.communities_coordinates = None
        self.edges = None
        self.nodes = None
        self.all_primitives = None
        self.nodes_communities = None

        self.first_step_detection_config = asdict(first_step_detection_config)
        self.rest_steps_detection_config = asdict(rest_steps_detection_config)
        self.n_iter = n_iter

        self._communities_generator = _ClusterHulls(points)

        self.communities_info = self._communities_generator.get_communities(
            points,
            detection_method,
            first_step_detection_config=self.first_step_detection_config,
            rest_steps_detection_config=self.rest_steps_detection_config,
            n_iter=self.n_iter,
            **kwargs,
        )

        self._sort_and_filter_communities()

    def _sort_and_filter_communities(self):
        """Sorts and filters communities for easier maintenance."""
        filtered_communities_points = [
            community
            for community in self.communities_info["points"]
            if len(community) > 1
        ]

        sorted_communities_points = sorted(
            filtered_communities_points, key=lambda x: (len(x), x[0])
        )
        self.communities = [
            frozenset(community) for community in sorted_communities_points
        ]

        filtered_and_sorted_community_pairs = sorted(
            (
                (tc, tp)
                for tc, tp in zip(
                    self.communities_info["vertices"], self.communities_info["points"]
                )
                if len(tp) > 1
            ),
            key=lambda x: (len(x[1]), x[1][0]),
        )

        self.communities_coordinates = [
            community_coo
            for community_coo, community_points in filtered_and_sorted_community_pairs
        ]

        self.edges = sorted(
            [frozenset(edge) for edge in self.local_graph.edges()],
            key=lambda x: list(x)[0],
        )

        self.nodes = sorted(
            [frozenset([node]) for node in self.local_graph.nodes()],
            key=lambda x: list(x)[0],
        )
        self.all_primitives = self.nodes + self.edges + self.communities
        self.nodes_communities = self.nodes + self.communities

    @property
    def primitives(self):
        """Returns a list of all nodes and communities."""
        return self.nodes_communities

    def get_skeleton(self, k):
        """Returns the k-skeleton of the community graph.

        Args:
            k (int): The dimension of the skeleton.

        Returns:
            list: The k-skeleton of the community graph.
        """
        return [
            primitive for primitive in self.all_primitives if len(primitive) - 1 <= k
        ]

    @property
    def lengths(self):
        """Returns a tuple of the number of nodes, edges, and communities."""
        return len(self.nodes), len(self.edges), len(self.communities)


class _ClusterHulls:
    def __init__(self, points, **kwargs):
        self.points = points

        self.index_map = {}
        self.point_to_original_index_map = {
            tuple(point): idx for idx, point in enumerate(self.points)
        }
        for new_idx, point in enumerate(self.points):
            original_idx = self.point_to_original_index_map[tuple(point)]
            self.index_map[new_idx] = original_idx

    def filter_valid_params(self, cls, config):
        valid_params = inspect.signature(cls.__init__).parameters
        return {k: v for k, v in config.items() if k in valid_params}
    
    def detection(
        self,
        points: np.ndarray,
        method: str = "optics",
        config: Dict[str, Union[int, float]] = None,
    ) -> List[int]:
        """Perform clustering on a set of points using the specified method.

        Args:
            points (np.ndarray): Array of points to cluster.
            method (str): Clustering method to use. Supported methods: 'optics', 'hdbscan', 'dbscan', 'spectral_clustering'.
            config (Dict[str, Union[int, float]]): Parameters for the clustering algorithm.

        Returns:
            List[int]: The cluster labels for each point.

        Raises:
            ValueError: If an unknown clustering method is specified.
        """
        if config is None:
            config = {}

        logging.info(f"Starting {method} clustering with configuration: {config}")
        if method == "optics":
            clusterer = OPTICS(**self.filter_valid_params(OPTICS, config))
        elif method == "hdbscan":
            clusterer = HDBSCAN(**self.filter_valid_params(HDBSCAN, config))
        elif method == "dbscan":
            clusterer = DBSCAN(**self.filter_valid_params(DBSCAN, config))
        elif method == "spectral_clustering":
            clusterer = SpectralClustering(**self.filter_valid_params(SpectralClustering, config))
        else:
            raise ValueError(f"Unknown clustering method specified: {method}")

        if method in ["optics", "hdbscan", "dbscan"]:
            cluster_labels = clusterer.fit_predict(points)
        elif method == "spectral_clustering":
            cluster_labels = clusterer.fit(points).labels_

        logging.info(
            f"Done with {method} clustering. Number of clusters found: {len(set(cluster_labels))}"
        )
        return cluster_labels

    # def detection(self, config):
    #     method = config.get('method')
    #     min_samples = config.get('min_samples')
    #     xi = config.get('xi')
    #     min_cluster_size = config.get('min_cluster_size')
    #     points_to_cluster = config.get('points', self.points)
    #     max_eps = config.get('max_eps')

    #     logging.info(f"Starting {method} clustering...")
    #     if method == "optics":
    #         clusterer = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, max_eps=max_eps)
    #     elif method == "hdbscan":
    #         clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    #     elif method == "dbscan":
    #         clusterer = DBSCAN(min_samples=min_samples)
    #     elif method == "spectral_clustering":
    #         clusterer = SpectralClustering(n_clusters=min_cluster_size)
    #     else:
    #         raise ValueError("Unknown clustering method specified.")

    #     if method in ["optics", "hdbscan", "dbscan"]:
    #         cluster_labels = clusterer.fit_predict(points_to_cluster)
    #     elif method == "spectral_clustering":
    #         cluster_labels = clusterer.fit(points_to_cluster).labels_

    #     logging.info(f"Done with {method} clustering.")
    #     return cluster_labels

    def get_communities(
        self, points, method, first_step_detection_config, rest_steps_detection_config, n_iter
    ):

        initial_labels = self.detection(points, method, first_step_detection_config)
        labels = initial_labels
        if n_iter > 0:
            logging.info("Extending the detection...")
            for _ in range(n_iter):
                if rest_steps_detection_config and -1 in labels:
                    refinement_indices = np.where(labels == -1)[0]
                    refinement_points = self.points[refinement_indices]
                    refinement_labels = self.detection(
                        refinement_points, method, rest_steps_detection_config
                    )

                    max_label = max(labels.max(), 0)
                    remapped_refinement_labels = np.where(
                        refinement_labels != -1, refinement_labels + max_label + 1, -1
                    )
                    for idx, new_label in zip(
                        refinement_indices, remapped_refinement_labels
                    ):
                        labels[idx] = new_label

        communities = {"vertices": [], "points": []}
        for label in np.unique(labels):
            label_indices = np.where(labels == label)[0]
            original_indices = [
                self.index_map[idx] for idx in label_indices
            ]  # Convert to original indices
            cluster_points = self.points[label_indices]
            try:
                if len(cluster_points) >= 3:
                    hull = ConvexHull(cluster_points)
                    hull_points = hull.points[hull.vertices]
                    communities["vertices"].append(
                        np.array([]) if label == -1 else hull_points
                    )
                    communities["points"].append(original_indices)
            except Exception as e:
                logging.error(
                    f"At least one community failed to be constructed due to: {e}. Try increasing 'min_samples' or 'min_clusters'."
                )

        return communities


def separate_configurations(config):

    first_step_detection_config = {}
    rest_steps_detection_config = {}

    for key, value in config.items():
        if key.startswith("initial"):
            new_key = key.replace("initial_", "")
            first_step_detection_config[new_key] = value
        elif key.startswith("refinement"):
            new_key = key.replace("refinement_", "")
            rest_steps_detection_config[new_key] = value

    return first_step_detection_config, rest_steps_detection_config


# def create_graph_from_point_cloud(point_cloud, radius):
#     model = NearestNeighbors(radius=radius)
#     model.fit(point_cloud)

#     edges = []
#     for idx, point in enumerate(point_cloud):
#         indices = model.radius_neighbors([point], return_distance=False)
#         edges.extend([(idx, i) for i in indices[0] if i != idx])

#     # Create a graph from the edges
#     G = nx.Graph()
#     G.add_edges_from(edges)

#     for idx in range(len(point_cloud)):
#         if idx not in G:
#             G.add_node(idx)

#     return G


# def create_graph_from_point_cloud(point_cloud, radius, k=None):
#     model = NearestNeighbors(radius=radius)
#     model.fit(point_cloud)

#     edges = []
#     if k is not None:
#         # If k is specified, find up to k neighbors within the given radius
#         distances, indices = model.radius_neighbors(point_cloud, radius, sort_results=True)
#         for idx, neighbors in enumerate(indices):
#             for i, neighbor_idx in enumerate(neighbors):
#                 if neighbor_idx != idx and i < k:  # Check if within k nearest neighbors
#                     edges.append((idx, neighbor_idx))
#     else:
#         # If k is not specified, only consider the radius for creating edges
#         indices = model.radius_neighbors(point_cloud, radius, return_distance=False)
#         for idx, neighbors in enumerate(indices):
#             edges.extend([(idx, neighbor_idx) for neighbor_idx in neighbors if neighbor_idx != idx])

#     # Create a graph from the edges
#     G = nx.Graph()
#     G.add_edges_from(edges)

#     # Ensure all points are included in the graph as nodes
#     for idx in range(len(point_cloud)):
#         if idx not in G:
#             G.add_node(idx)

#     return G


def create_graph_from_point_cloud(point_cloud, radius=None, k=None):
    if radius is None and k is None:
        from scipy.spatial import Delaunay

        tri = Delaunay(point_cloud)
        edges = set(
            (min(a, b), max(a, b))
            for simplex in tri.simplices
            for a, b in zip(simplex, simplex[1:])
        )
    else:
        model = NearestNeighbors(
            n_neighbors=min(k, len(point_cloud) - 1) if k else None,
            radius=radius if radius else None,
        )
        model.fit(point_cloud)
        edges = set()

        if k is not None:
            distances, indices = model.kneighbors(point_cloud)

            if radius is not None:
                # Only add edges if within the radius
                for idx, (dist, neighs) in enumerate(zip(distances, indices)):
                    for d, n in zip(dist, neighs):
                        if d <= radius and idx != n:
                            edges.add((min(idx, n), max(idx, n)))
            else:
                # Add all k-nearest neighbors
                for idx, neighs in enumerate(indices):
                    edges.update((min(idx, n), max(idx, n)) for n in neighs if idx != n)
        else:
            # Radius only, get all neighbors within the radius
            distances, indices = model.radius_neighbors(point_cloud)
            for idx, neighs in enumerate(indices):
                edges.update((min(idx, n), max(idx, n)) for n in neighs if idx != n)

    G = nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from(range(len(point_cloud)))
    return G
