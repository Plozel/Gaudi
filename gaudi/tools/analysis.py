import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import scanpy as sc


def generate_labels(
    sample,
    embeddings_key: str = None,
    level: int = 1,
    clustering_method: str = "kmeans",
    layer: str = None,
    labels_key: str = "gaudi_labels",
    n_clusters: int = None,
    resolution: float = 0.5,
    neighbors_metric: str = "cosine",
    n_neighbors: int = 30,
    seed: int = 1,
    **kwargs,
):

    if embeddings_key is not None and embeddings_key in sample.perspectives.keys():
        data = sample.perspectives[embeddings_key][level].copy()
    elif sample.embeds[level] is not None:
        data = sc.AnnData(sample.embeds[level]).copy()
    else:
        raise ValueError(
            "Embeddings not found in the sample. Please use the get_representations function to generate embeddings first, or provide a valid data type name for the embeddings_key parameter."
        )

    if clustering_method == "kmeans":
        if n_clusters is None:
            raise ValueError(
                "Number of clusters must be specified for kmeans clustering."
            )

        if layer is not None:
            if layer not in data.layers.keys():
                logging.warning(
                    f"Layer {layer} not found in the data. Using the default data matrix instead."
                )
                matrix_data = data.X
            else:
                matrix_data = data.layers[layer]
        else:
            matrix_data = data.X

        clustering_model = KMeans(n_clusters=n_clusters, random_state=seed, **kwargs)
        cluster_assignments = clustering_model.fit_predict(matrix_data, **kwargs)
    elif clustering_method == "leiden":

        use_rep = layer if layer and layer in data.layers.keys() else "X"
        data.obsm[use_rep] = data.layers[use_rep] if use_rep in data.layers.keys() else data.X
        
        sc.pp.neighbors(
            data,
            n_neighbors=n_neighbors,
            use_rep=use_rep,
            metric=neighbors_metric,
            **kwargs,
        )
        sc.tl.leiden(data, resolution=resolution, **kwargs)
        cluster_assignments = data.obs["leiden"].values
    else:
        raise ValueError("Invalid clustering method specified.")

    cluster_assignments = cluster_assignments.astype("str")
    # Create a dictionary with cluster assignments
    cluster_dict = {labels_key: cluster_assignments}

    # Store the cluster assignments in the sample's labels list at the specified order level
    if sample.labels[level] is None:
        sample.labels[level] = {}
    sample.labels[level].update(cluster_dict)


def compare(
    adata,
    groupby: str,
    groups="all",
    reference="rest",
    method: str = "wilcoxon",
    layer=None,
    min_in_group_fraction=0.25,
    min_fold_change=0.2,
    max_out_group_fraction=1.1,
    compare_abs=False,
    tie_correct=True,
    n_features=10,
    key="ranked_features_groups",
    diff_pct=None,
    min_counts=1,
    exclude_labels=None,
    max_pvalue=0.05,
    **kwargs,
) -> None:

    labels = adata.obs[groupby]
    valid_labels = _get_valid_labels(labels, min_counts, exclude_labels)

    if groups != "all":
        groups = set(groups).intersection(valid_labels)
    else:
        groups = valid_labels

    if len(groups) < 2:
        raise ValueError("At least two groups are required for comparison.")

    adata = adata[adata.obs[groupby].isin(groups)]

    sc.tl.rank_genes_groups(
        adata,
        groupby,
        groups=groups,
        reference=reference,
        method=method,
        key_added=key,
        layer=layer,
        pts=True,
        tie_correct=tie_correct,
        **kwargs,
    )

    key_filtered = f"{key}_filtered"
    sc.tl.filter_rank_genes_groups(
        adata,
        key=key,
        key_added=key_filtered,
        min_in_group_fraction=min_in_group_fraction,
        min_fold_change=min_fold_change,
        max_out_group_fraction=max_out_group_fraction,
        compare_abs=compare_abs,
        **kwargs,
    )

    if diff_pct is not None:
        significant_genes = adata.uns[key_filtered]["names"]
        # Create a copy of significant_genes to hold the filtered genes
        filtered_genes_array = np.copy(significant_genes)
        # Get the number of clusters
        num_clusters = len(significant_genes.dtype.names)
        # Iterate through each gene (tuple of gene names)
        for gene_idx, gene_tuple in enumerate(significant_genes):
            # Iterate through each cluster
            for cluster_idx in range(num_clusters):
                gene = gene_tuple[cluster_idx]
                if pd.isna(
                    gene
                ):  # Check if gene is nan and skip to next iteration if true
                    continue
                cluster_label = significant_genes.dtype.names[cluster_idx]
                cluster_non_zero_pct = adata.uns[key_filtered]["pts"].loc[
                    gene, cluster_label
                ]
                other_clusters_non_zero_pct = adata.uns[key_filtered]["pts_rest"].loc[
                    gene, cluster_label
                ]
                pct_diff = cluster_non_zero_pct - other_clusters_non_zero_pct
                if pct_diff < diff_pct:
                    # If the percentage difference is less than diff_pct, set the gene to nan in the filtered array
                    filtered_genes_array[gene_idx][cluster_idx] = np.nan

        if max_pvalue is not None:
            significant_pvalues = adata.uns[key_filtered][
                "pvals_adj"
            ]  # Adjusted p-values
            # No need to copy the filtered_genes_array again
            num_clusters = len(significant_pvalues.dtype.names)
            for gene_idx, gene_tuple in enumerate(significant_pvalues):
                for cluster_idx in range(num_clusters):
                    pvalue = gene_tuple[cluster_idx]
                    if (
                        pvalue > max_pvalue
                        and filtered_genes_array[gene_idx][cluster_idx] is not np.nan
                    ):
                        # If the p-value exceeds max_pvalue, set the gene to nan in the filtered array
                        filtered_genes_array[gene_idx][cluster_idx] = np.nan

        # Update the entry in adata.uns
        adata.uns[key_filtered]["names"] = filtered_genes_array

    filtered_genes = adata.uns[key_filtered]["names"]

    # Convert the structured array to a DataFrame
    filtered_genes_df = pd.DataFrame(filtered_genes)
    if n_features is not None:
        filtered_genes_df = filtered_genes_df.head(n_features)
    non_nan_values = []
    subsets_dict = {}
    for column in filtered_genes_df.columns:
        non_nan_values_col = filtered_genes_df[column].dropna().tolist()
        if len(non_nan_values_col) == 0:
            continue
        non_nan_values.extend(non_nan_values_col)
        subsets_dict[column] = non_nan_values_col

    return non_nan_values, subsets_dict


def _get_valid_labels(labels, min_counts=1, exclude_labels=None):

    unique_labels, counts = np.unique(labels, return_counts=True)
    less_than_min_counts_labels = set(unique_labels[counts <= min_counts])
    unique_labels = set(unique_labels)
    valid_labels = unique_labels.difference(less_than_min_counts_labels)
    if exclude_labels is not None:
        valid_labels = valid_labels.difference(exclude_labels)
        print(
            f"Excluded labels: {exclude_labels}. Reason: These labels were excluded by the user."
        )

    if len(less_than_min_counts_labels) > 0:
        print(
            f"Excluded labels: {less_than_min_counts_labels}. Reason: These labels represent clusters with less than {min_counts + 1} observations."
        )

    return valid_labels
