import numpy as np
from pathlib import Path
import pandas as pd
import scanpy as sc

EPS = 1e-10

def files_exist(files):
    return all([Path(f).exists() for f in files])

# def sparse_average_covariance(X, rowvar=False, axis=None, metric="covariance"):
#     """
#     Compute the average covariance or correlation for a sparse matrix X.
#     If rowvar is False, each column represents a variable, with observations in the rows.
    
#     Parameters:
#     - X: A scipy.sparse matrix in CSR format.
#     - rowvar: If False, each column represents a variable, with observations in the rows.
#     - metric: "covariance" or "correlation" - specifies which metric to compute and return.
    
#     Returns:
#     - Corrected average metric (covariance or correlation) as a scalar.
#     """
#     # Transpose X if rowvar is True to ensure columns represent variables
#     if rowvar:
#         X = X.T
    
#     n_samples = X.shape[0]

#     # Subtract the mean from each column (variable)
#     col_means = X.mean(axis=0)
#     X_centered = X - col_means

#     # Compute covariance matrix
#     cov_matrix = X_centered.T.dot(X_centered) / (n_samples - 1)

#     if metric == "covariance":
#         # Calculate the average covariance
#         if axis is not None:
#             return cov_matrix.mean(axis=axis)
#         else:
#             return cov_matrix.mean()
#     elif metric == "correlation":
#         # Compute standard deviations for each column
#         variances = cov_matrix.diagonal()
#         std_devs = np.sqrt(variances + EPS)
        
#         # Compute correlation matrix using broadcasting
#         corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        
#         # Calculate the average correlation
#         if axis is not None:
#             return corr_matrix.mean(axis=axis)
#         else:
#             return corr_matrix.mean()
#     elif metric == "variance":
#         variances = cov_matrix.diagonal()
#         return variances
#     else:
#         raise ValueError("metric must be 'covariance' or 'correlation'")


def read_and_save_xenium_data(cells_info_path, cell_feature_matrix_path, output_h5ad_path):
    """
    Read Xenium data and save it as an AnnData object.

    Example: 
    cells_info_path =  "cells.csv.gz"
    cell_feature_matrix_path = "cell_feature_matrix.h5"
    output_h5ad_path = "adata.h5ad"
    read_and_save_xenium_data(cells_info_path, cell_feature_matrix_path, output_h5ad_path)
    """

    # Read cell information from the compressed CSV file
    cells_info = pd.read_csv(cells_info_path, compression='gzip', index_col=0)
    coordinates = cells_info[['x_centroid', 'y_centroid']].values

    # Read 10x data into a Scanpy AnnData object
    adata = sc.read_10x_h5(cell_feature_matrix_path)

    # Add spatial coordinates to the AnnData object
    adata.obsm['spatial'] = coordinates

    # Save the AnnData object as an h5ad file
    adata.write_h5ad(output_h5ad_path)
