import logging
import scanpy as sc
import numpy as np

logging.basicConfig(level=logging.INFO)


def filter_data_no_sc(
    adata,
    min_counts=0,
    max_counts=float("inf"),
    min_genes=0,
    max_pct_mt=float("inf"),
    min_cells=0,
    is_there_mt_genes=False,
):
    """Filter data based on cell counts, gene counts, and mitochondrial gene expression with no reliance on scanpy."""
    counts_per_cell = np.sum(adata.X, axis=1)
    genes_per_cell = np.sum(adata.X > 0, axis=1)

    cell_filter_mask = (
        (counts_per_cell >= min_counts)
        & (counts_per_cell <= max_counts)
        & (genes_per_cell >= min_genes)
    )

    adata = adata[cell_filter_mask]

    if is_there_mt_genes:
        mt_gene_mask = adata.var_names.str.startswith("MT-")
        counts_mt_per_cell = np.sum(adata[:, mt_gene_mask].X, axis=1)
        pct_counts_mt = (counts_mt_per_cell / counts_per_cell) * 100

        adata = adata[pct_counts_mt < max_pct_mt]

    cells_per_gene = np.sum(adata.X > 0, axis=0)
    gene_filter_mask = cells_per_gene >= min_cells

    adata = adata[:, gene_filter_mask]

    logging.info(f"#Cells after filtering: {adata.n_obs}")
    logging.info(f"#Genes after filtering: {adata.n_vars}")

    return adata


def run_preprocessing(
    adata,
    pca_components=100,
    min_counts=10,
    max_counts=None,
    min_cells=30,
    min_genes=10,
    max_pct_mt=20,
    n_top_genes=None,
):
    """Run standard preprocessing pipeline for spatial transcriptomics data.

    Args:
        adata (anndata.AnnData): Annotated data matrix.
        pca_components (int, optional): Number of principal components. Defaults to 100.
        min_counts (int, optional): Minimum total counts per cell. Defaults to 10.
        max_counts (int, optional): Maximum total counts per cell. Defaults to None.
        min_cells (int, optional): Minimum number of cells a gene must be expressed in. Defaults to 30.
        min_genes (int, optional): Minimum number of expressed genes per cell. Defaults to 10.
        max_pct_mt (float, optional): Maximum percentage of mitochondrial gene expression per cell. Defaults to 20.
        n_top_genes (int, optional): Number of highly variable genes to select. Defaults to 4000.

    Returns:
        anndata.AnnData: Preprocessed annotated data matrix.
    """
    logging.info("Starting quality control and preprocessing of anndata...")
    logging.info(f"#Cells before filtering: {adata.n_obs}")
    logging.info(f"#Genes before filtering: {adata.n_vars}")

    adata.var_names = [gene_name.upper() for gene_name in adata.var_names]
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")

    is_there_qc_matrices = False
    try:
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        is_there_mt_genes = True
        is_there_qc_matrices = True
    except Exception as e:
        logging.warning(
            f"Error calculating QC metrics: {e}. Proceeding without mitochondrial genes."
        )
        is_there_mt_genes = False
        try:
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            is_there_qc_matrices = True
        except Exception as e:
            logging.warning(
                f"Error calculating QC metrics: {e}. Proceeding without QC metrics."
            )
            is_there_qc_matrices = False

    if is_there_qc_matrices:
        sc.pp.filter_cells(adata, min_counts=min_counts)
        if max_counts is not None:
            sc.pp.filter_cells(adata, max_counts=max_counts)
        sc.pp.filter_cells(adata, min_genes=min_genes)

        if is_there_mt_genes:
            adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt]
            adata = adata[:, ~adata.var["mt"].values]

        sc.pp.filter_genes(adata, min_cells=min_cells)
    else:
        adata = filter_data_no_sc(
            adata,
            min_counts=min_counts,
            max_counts=max_counts,
            min_genes=min_genes,
            max_pct_mt=max_pct_mt,
            min_cells=min_cells,
            is_there_mt_genes=is_there_mt_genes,
        )
    
    logging.info(f"#Cells after filtering: {adata.n_obs}")
    logging.info(f"#Genes after filtering: {adata.n_vars}")

    logging.info("Normalizing data...")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    adata.layers["log_normalized"] = adata.X.copy()

    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    sc.pp.scale(adata)
    adata.layers["scaled_log_normalized"] = adata.X.copy()
    logging.info("Performing PCA...")
    sc.pp.pca(adata, n_comps=pca_components)

    adata.X = adata.layers["log_normalized"].copy()
    adata.uns["preprocessing_flag"] = True
    logging.info("Preprocessing completed.")
    logging.info("anndata object X matrix is now log normalized")
    logging.info(
        "anndata object original counts and normalized counts are stored in adata.layers['counts'] and adata.layers['log_normalized'] respectively"
    )

    return adata.copy()
