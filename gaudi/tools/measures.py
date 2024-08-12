import os
import logging
import pandas as pd
from scipy.sparse import hstack, csr_matrix, vstack, issparse, lil_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import scanpy as sc
import anndata as ad
import torch
from tqdm import tqdm

EPS = 1e-10


def aggregate_data(data, method):
    """
    Aggregates the data based on the specified method.

    Args:
        data: The data to aggregate.
        method (str): The type of aggregation ('mean', 'max', 'var', 'median', etc.).

    Returns:
        Aggregated data.
    """
    if method == "mean":
        return data.mean(axis=0)
    elif method == "max_ratio":
        agg_data = data.max(axis=0)
        return agg_data / (np.median(data, axis=0) + EPS)
    elif method == "var":
        return data.var(axis=0)
    elif method == "median":
        return np.median(data, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")


def compute_basic_measure(sample, aggregation_type="mean", level=1, main_composition_key=None):
    """
    Aggregates cell data globally and per group based on the specified aggregation type.

    Parameters:
    sample: An object containing AnnData.
    aggregation_type (str): The type of aggregation to perform ('mean', 'max', 'variance', etc.).
    level (int): The granularity level for aggregation.
    main_composition_key (str, optional): Key in sample.base_adata.obs used for grouping.
    """

    init_log_normalized = (
        csr_matrix(sample.base_adata.layers["log_normalized"])
        if issparse(sample.base_adata.layers["log_normalized"])
        else csr_matrix(sample.base_adata.layers["log_normalized"])
    )
    init_counts = (
        csr_matrix(sample.base_adata.layers["counts"])
        if issparse(sample.base_adata.layers["counts"])
        else csr_matrix(sample.base_adata.layers["counts"])
    )
    var_names = sample.base_adata.var_names
    num_level_primitives = sample.separators[level + 1] - sample.separators[level]

    # Initialize aggregation storage
    aggregated_data = {"global": {"log_normalized": lil_matrix((num_level_primitives, init_log_normalized.shape[1])),
                                  "counts": lil_matrix((num_level_primitives, init_counts.shape[1]))}}

    if main_composition_key:
        groups = sample.base_adata.obs[main_composition_key].unique()
        for group in groups:
            aggregated_data[group] = {"log_normalized": lil_matrix((num_level_primitives, init_log_normalized.shape[1])),
                                      "counts": lil_matrix((num_level_primitives, init_counts.shape[1]))}

    # Aggregate data for each primitive
    for primitive_id, primitive in enumerate(sample.primitives[sample.separators[level]:sample.separators[level + 1]]):
        indices = list(primitive)
        log_normalized_primitive = init_log_normalized[indices]
        counts_primitive = init_counts[indices]

        # Apply global aggregation
        aggregated_data["global"]["log_normalized"][primitive_id, :] = aggregate_data(log_normalized_primitive, aggregation_type)
        aggregated_data["global"]["counts"][primitive_id, :] = aggregate_data(counts_primitive, aggregation_type)

        # Apply group-specific aggregation if required
        if main_composition_key:
            for group in groups:
                group_indices = np.where(sample.base_adata.obs[main_composition_key] == group)[0]
                group_primitive_indices = [idx for idx in primitive if idx in group_indices]

                if group_primitive_indices:
                    group_log_normalized = init_log_normalized[group_primitive_indices]
                    group_counts = init_counts[group_primitive_indices]
                    aggregated_data[group]["log_normalized"][primitive_id, :] = aggregate_data(group_log_normalized, aggregation_type)
                    aggregated_data[group]["counts"][primitive_id, :] = aggregate_data(group_counts, aggregation_type)

    # Convert to dense format, then back to CSR, and store in an AnnData object
    obs = pd.DataFrame(index=[f"primitive_{i}" for i in range(num_level_primitives)])
    adata = ad.AnnData(X=aggregated_data["global"]["counts"].todense(), obs=obs, var=pd.DataFrame(index=var_names), dtype='float32')
    
    for key, value in aggregated_data.items():
        if key == "global":
            adata.layers[f"counts_based"] = csr_matrix(np.nan_to_num(value["counts"].todense(), nan=0.0))
            adata.layers[f"log_normalized_based"] = csr_matrix(np.nan_to_num(value["log_normalized"].todense(), nan=0.0))
        else:    
            adata.layers[f"{key}_counts_based"] = csr_matrix(np.nan_to_num(value["counts"].todense(), nan=0.0))
            adata.layers[f"{key}_log_normalized_based"] = csr_matrix(np.nan_to_num(value["log_normalized"].todense(), nan=0.0))

    return adata


def compute_basic_interactions(sample, cellchatdb_path):
    """
    Calculates ligand-receptor based pathways scores for each edge in the dataset.
    """

    tools_dir_path = os.path.dirname(os.path.realpath(__file__))
    gaudi_dir_path = os.path.abspath(os.path.join(tools_dir_path, '..'))
    file_path = os.path.join(gaudi_dir_path, cellchatdb_path)
    lr_db = pd.read_csv(file_path)
    lr_db = lr_db.applymap(lambda s: s.upper() if type(s) == str else s)

    lr_db[["ligand", "receptor"]] = lr_db["interaction_name"].str.split(
        "_", n=1, expand=True
    )

    logging.info("Getting interactions for each pair of neighbored 0-dim simplices...")

    adata = sample.base_adata.copy()
    valid_lr_db = _get_relevant_lrs(adata, lr_db)
    if valid_lr_db.empty:
        logging.warning(f"No relevant ligand-receptor pairs were found.")
        return

    (
        ligand_counts,
        receptor_counts,
        ligand_names,
        receptor_names,
        pathway_names,
    ) = _get_lr_data(valid_lr_db, adata)

    ligand_x, receptor_x, _, _, _ = _get_lr_data(valid_lr_db, adata, layer="X")

    ligand_counts = hstack(ligand_counts)
    receptor_counts = hstack(receptor_counts)
    ligand_x = hstack(ligand_x)
    receptor_x = hstack(receptor_x)

    adjacency_coo = adata.obsp["spatial_connectivities"].tocoo()
    edge_names = pd.Index(
        list(
            map(
                lambda x: f"{x[0]}_<->_{x[1]}",
                map(
                    sorted,
                    zip(
                        adata.obs_names[adjacency_coo.row],
                        adata.obs_names[adjacency_coo.col],
                    ),
                ),
            )
        )
    )

    lr_counts_scores_df_undirected = _compute_interactions_scores(
        ligand_counts[adjacency_coo.row],
        receptor_counts[adjacency_coo.col],
        edge_names,
    )
    lr_x_scores_df_undirected = _compute_interactions_scores(
        ligand_x[adjacency_coo.row], receptor_x[adjacency_coo.col], edge_names
    )

    lr_x_scores = csr_matrix(vstack(lr_x_scores_df_undirected["lr_scores"].values))
    lr_counts_scores = csr_matrix(
        vstack(lr_counts_scores_df_undirected["lr_scores"].values)
    )

    unique_pathway_names = np.unique(pathway_names)
    pathways_counts_scores = np.zeros(
        (lr_counts_scores.shape[0], len(unique_pathway_names))
    )
    pathways_x_scores = np.zeros((lr_x_scores.shape[0], len(unique_pathway_names)))

    for i, pathway_name in enumerate(unique_pathway_names):
        pathway_idx = np.where(np.array(pathway_names) == pathway_name)[0]
        pathways_x_scores[:, i] = np.array(
            lr_x_scores[:, pathway_idx].max(axis=1).todense().squeeze(1).tolist()[0]
        )
        pathways_counts_scores[:, i] = np.array(
            lr_counts_scores[:, pathway_idx]
            .max(axis=1)
            .todense()
            .squeeze(1)
            .tolist()[0]
        )

    non_zero_cols = np.unique(pathways_x_scores.nonzero()[1])
    pathways_x_scores = pathways_x_scores[:, non_zero_cols]
    pathways_counts_scores = pathways_counts_scores[:, non_zero_cols]
    var = pd.DataFrame(index=unique_pathway_names[non_zero_cols])
    obs = pd.DataFrame(index=lr_counts_scores_df_undirected.index)
    lr_based_pathways_adata = sc.AnnData(X=pathways_x_scores, var=var, obs=obs)
    lr_based_pathways_adata.layers["counts_based"] = pathways_counts_scores

    lr_based_pathways_adata.layers["log_normalized_based"] = csr_matrix(pathways_x_scores.copy())
    lr_based_pathways_adata.layers["counts_based"] = csr_matrix(pathways_counts_scores.copy())
    lr_based_pathways_adata.X = csr_matrix(pathways_counts_scores.copy())
    return lr_based_pathways_adata



def _get_relevant_lrs(adata, lr_db):
    """Filter relevant ligand-receptor pairs which are present in the AnnData."""

    var_names_set = set(adata.var_names)
    var_names_set = [gene.upper() for gene in var_names_set]

    valid_lr_db = lr_db[
        lr_db.apply(
            lambda row: row["ligand"] in var_names_set
            and any(
                receptor in var_names_set for receptor in row["receptor"].split("_")
            ),
            axis=1,
        )
    ]

    return valid_lr_db


def _get_lr_data(valid_lr_db, adata, layer="counts"):
    """Extract expression data and names for relevant ligand-receptor pairs."""

    ligand_data = []
    receptor_data = []
    ligand_names = []
    receptor_names = []
    pathway_names = []
    var_names_set = list(adata.var_names)
    var_names_set = [gene.upper() for gene in var_names_set]
    adata.var_names = var_names_set

    for _, row in tqdm(valid_lr_db.iterrows()):
        ligand = row["ligand"].upper()
        receptors_in_group = row["receptor"].split("_")
        receptors_in_group = [receptor.upper() for receptor in receptors_in_group]
        valid_receptors_in_group = [
            receptor for receptor in receptors_in_group if receptor in var_names_set
        ]

        if valid_receptors_in_group:
            ligand_expr_data = adata[:, ligand]
            if layer != "X":
                ligand_expr = ligand_expr_data.layers[layer]
                receptor_expr_list = [
                    adata[:, receptor].layers[layer].toarray()
                    for receptor in valid_receptors_in_group
                ]
            else:
                ligand_expr = ligand_expr_data.X
                receptor_expr_list = [
                    adata[:, receptor].X.toarray()
                    for receptor in valid_receptors_in_group
                ]
            receptor_expr = np.max(receptor_expr_list, axis=0)

            ligand_data.append(ligand_expr)
            receptor_data.append(csr_matrix(receptor_expr))
            ligand_names.append(ligand)
            receptor_names.append(row["receptor"])
            pathway_names.append(row["pathway_name"])
        else:
            raise ValueError(
                "Something is wrong...At least one receptor should have been found."
            )

    return ligand_data, receptor_data, ligand_names, receptor_names, pathway_names


def _compute_interactions_scores(ligands_nodes, receptors_nodes, edge_names):
    """Compute ligand-receptor interaction scores and aggregate the scores dataframe."""

    lr_scores_directed = ligands_nodes.multiply(receptors_nodes)
    lr_scores_df_directed = pd.DataFrame(
        {"lr_scores": lr_scores_directed}, index=edge_names
    )
    lr_scores_df_undirected = lr_scores_df_directed.groupby(
        lr_scores_df_directed.index
    ).agg(lambda x: csr_matrix.maximum(*x))

    return lr_scores_df_undirected




def compute_higher_order_interactions(sample, data):

    level = 1
    primitive_to_idx = sample.primitive_to_idx

    community_edge_index = data.community_edge_index
    edge_index = data.edge_index

    separators = sample.separators
    ids = np.array(range(separators[level], separators[level + 1]))
    pos = data.pos
    zero_obs_names = np.array(sample.base_adata.obs_names)
    if sample.perspectives["lr_pathways"][0] is not None:
        adata_lr = sample.perspectives["lr_pathways"][0]
    else:
        logging.warning(
            f"Can't find calculated interactions at level 0. Skipping..."
        )
        return

    tiles = data.primitives[separators[level] :]

    # Initialize result arrays with zeros
    higher_order_lr_counts_scores = np.zeros((len(tiles), adata_lr.shape[1]))
    higher_order_lr_x_scores = np.zeros((len(tiles), adata_lr.shape[1]))

    # Preprocessing: Transform adata_lr to dictionary for faster access
    adata_lr_dict = {
        name: (adata_lr[name, :].X, adata_lr[name, :].layers["counts_based"])
        for name in adata_lr.obs_names
    }

    # Iterate through the simplices
    for i, tile in enumerate(tqdm(tiles)):
        nodes = community_edge_index[
            0, community_edge_index[1, :] == primitive_to_idx[tile]
        ]

        # Check if the nodes of each edge are in the nodes tensor
        mask_src = torch.isin(edge_index[0], nodes)
        mask_dst = torch.isin(edge_index[1], nodes)

        # An edge is induced if both its nodes are in the nodes tensor
        mask_induced = mask_src & mask_dst

        # Gather the induced edges
        induced_edges = edge_index[:, mask_induced].numpy().T

        for edge in induced_edges:
            edge = tuple(edge)

            zero_simplices_0_names = [zero_obs_names[edge[0]]]
            zero_simplices_1_names = [zero_obs_names[edge[1]]]
            one_dim_simplices_names = [
                "_<->_".join(sorted(pair))
                for pair in zip(zero_simplices_0_names, zero_simplices_1_names)
            ]

            # Use dictionary for faster data access
            adata_scores, adata_counts = adata_lr_dict.get(
                one_dim_simplices_names[0], (0, 0)
            )

            if issparse(adata_scores):
                adata_scores = adata_scores.toarray()
            if issparse(adata_counts):
                adata_counts = adata_counts.toarray()
                
            higher_order_lr_x_scores[i, :] += adata_scores.squeeze()
            higher_order_lr_counts_scores[i, :] += adata_counts.squeeze()

    sorted_zero_obs_names = [
        "_<->_".join(sorted(zero_obs_names[list(tile)])) for tile in tiles
    ]
    lr_x_scores = csr_matrix(higher_order_lr_x_scores)
    lr_counts_scores = csr_matrix(higher_order_lr_counts_scores)

    adata = sc.AnnData(
        X=lr_counts_scores,
        var=pd.DataFrame(index=adata_lr.var_names),
        obs=pd.DataFrame(index=sorted_zero_obs_names),
    )

    adata.layers["counts_based"] = lr_counts_scores
    adata.layers["log_normalized_based"] = lr_x_scores

    return adata



def compute_higher_order_programs_scores(sample, geneset_db_path, level, use_mean=True):
    if not os.path.exists(geneset_db_path):
        print("No geneset database found. Skipping programs generation.")
        return
    programs_gene_sets = pd.read_csv(geneset_db_path, index_col=None)
    programs_gene_sets = programs_gene_sets.applymap(
        lambda s: s.upper() if type(s) == str else s
    )


    sample.perspectives["programs"][level] = []

    programs_dict = {}
    if use_mean:
        logging.info(
            f"Scoring {programs_gene_sets.shape[0]} genesets with mean scaled expression for Level {level}"
        )
    else:
        logging.info(f"Scoring {programs_gene_sets.shape[0]} genesets with sc.tl.score_genes for Level {level}")



    try:
        adata = sample.perspectives["mean"][level]
    except KeyError:
        logging.error(
            f"Can't find mean-based object at Level {level}. Skipping..."
        )


    if not isinstance(adata.X, csr_matrix):
        adata.X = csr_matrix(adata.X)

    var_names = adata.var_names

    X_dense = adata.X.toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense.T).T


    for row in tqdm(programs_gene_sets.itertuples(index=False)):
        if use_mean:
            data_to_process = (X_scaled, var_names, level, tuple(row))
        else:
            data_to_process = (adata, var_names, level, tuple(row))
        key, val = get_gene_set_score(data_to_process, use_mean=use_mean)
        if key is not None:
            programs_dict[key] = val

    x = np.stack(list(programs_dict.values())).T
    x[x < 0] = 0
    x = csr_matrix(x)
    obs = sample.perspectives["mean"][level].obs
    var = pd.DataFrame(index=list(programs_dict.keys()))
    adata = ad.AnnData(X=x, obs=obs, var=var, dtype="float32")
    adata.layers["counts_based"] = x
    adata.layers["log_normalized_based"] = x

    return adata



def get_gene_set_score(data_to_process, use_mean=True):
    if use_mean:
        X_scaled, var_names, level, row = data_to_process
    else:
        adata, var_names, level, row = data_to_process
    
    program_description, genes = row[0], row[1:]
    genes = [gene for gene in genes if gene in var_names]

    if len(genes) == 0:
        return None, None

    if use_mean:
        valid_genes_mask = var_names.isin(genes)
        if not valid_genes_mask.any():
            return None, None
        X_scaled_selected = X_scaled[:, valid_genes_mask]
        score = np.mean(X_scaled_selected, axis=1).squeeze()

        return program_description, score
    else:
        logging.info(f"Scoring geneset with sc.tl.score_genes for Level {level}")
        sc.tl.score_genes(
            adata,
            gene_list=genes,
            score_name="tmp_program_score",
            use_raw=False,
        )
        return program_description, adata.obs["tmp_program_score"].to_numpy()






def convert_discrit_to_composition(categories_df, prefix=None):
    """Converts a DataFrame with discrete categories to a composition DataFrame.
    
    Args:
        categories_df (pd.DataFrame): DataFrame with discrete categories.
        prefix (str, optional): Prefix to add to the columns of the composition DataFrame.
    """
    unique_categories = np.unique(categories_df.iloc[:, 0].values.astype('str'))
    num_unique_categories = len(unique_categories)
    categories_composition_per_cell = np.zeros(
        (categories_df.shape[0], num_unique_categories)
    )

    labels_to_idx = {label: i for i, label in enumerate(unique_categories)}

    for i, label in enumerate(categories_df.iloc[:, 0]):
        categories_composition_per_cell[i, labels_to_idx[str(label)]] = 1

    categories_composition_df = pd.DataFrame(
        categories_composition_per_cell, index=categories_df.index
    )

    categories_composition_df.columns = unique_categories
    if prefix is not None:
        categories_composition_df = categories_composition_df.add_prefix(prefix)

    return categories_composition_df


def compute_compositions(
        sample, group_1, group_2, group_1_level, group_2_level, group_1_level_prefix=None, group_2_level_prefix=None
    ):
    """
    Calculate the composition of group_2 categories per group_1 label and per community.

    Args:
        sample (HigherOrderData)
        group_1 (str): The group 1's labels name.
        group_2 (str): The group 2's labels name.
        group_1_level (int): The group 1 level.
        group_2_level (int): The group 2 level.
        group_1_level_prefix (str, optional): The group 1 level prefix.
        group_2_level_prefix (str, optional): The group 2 level prefix.
    """
    try:
        categories_df = pd.DataFrame(sample.labels[group_2_level][group_2])
    except KeyError:
        logging.error(f"Can't find the labels for group 2 at Level {group_2_level}. Check if {group_2} is in sample.labels[{group_2_level}] within the given gaudi object.")
        return
    
    # Convert discrete labels to composition
    categories_df = convert_discrit_to_composition(categories_df, prefix=group_2_level_prefix)

    # Scale the categories data frame
    categories_df_scaled = categories_df.div(categories_df.sum(axis=1), axis=0)

    separators = sample.separators
    tiles = sample.primitives[
        separators[group_1_level] : separators[group_1_level + 1]
    ]

    group_1_labels = sample.labels[group_1_level][group_1]
    unique_group_1_labels = sorted(np.unique(group_1_labels))

    # Initialize the mapping with zeros for each label and cell
    higher_dim_labels_to_0_dim_simplices = np.zeros((len(unique_group_1_labels), categories_df_scaled.shape[0]))

    # Create a mapping of higher dimension labels to zero dimension simplices
    for i, tile in enumerate(tiles):
        higher_dim_labels_to_0_dim_simplices[unique_group_1_labels.index(group_1_labels[i]), list(tile)] = 1

    # Transpose the mapping to align with the structure of celltypes_df_scaled
    zero_dim_simplices_to_higher_dim_labels_df = pd.DataFrame(
        higher_dim_labels_to_0_dim_simplices.T, columns=unique_group_1_labels, index=categories_df_scaled.index
    )

    # Initialize an empty DataFrame for the final distribution per label
    celltypes_distribution_per_label_df = pd.DataFrame(index=categories_df_scaled.columns)

    # Calculate the distribution for each label
    for label in unique_group_1_labels:
        label_cells = zero_dim_simplices_to_higher_dim_labels_df[zero_dim_simplices_to_higher_dim_labels_df[label] == 1].index
        if not set(label_cells).issubset(categories_df_scaled.index):
            raise ValueError("Indices mismatch between cell type and label dataframes.")
        label_distribution = categories_df_scaled.loc[label_cells].mean()
        celltypes_distribution_per_label_df[label] = label_distribution

    # Add prefix if provided
    if group_1_level_prefix is not None:
        celltypes_distribution_per_label_df = (
            celltypes_distribution_per_label_df.add_prefix(group_1_level_prefix)
        )

    # Initialize a dictionary for the final distribution per community (tile)
    community_distributions = {}

    # Calculate the distribution for each community (tile)
    for i, tile in enumerate(tiles):
        community_distribution = categories_df_scaled.loc[list(tile)].mean()
        community_distributions[f"community_{i}"] = community_distribution

    # Create DataFrame from community distributions dictionary
    community_distributions_df = pd.DataFrame(community_distributions)

    # Initialize AnnData object for community distributions
    community_distributions_adata = ad.AnnData(X=community_distributions_df.T)

    return celltypes_distribution_per_label_df, community_distributions_adata
