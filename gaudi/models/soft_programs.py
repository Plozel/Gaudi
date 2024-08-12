#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


EPS = 1e-10

class SoftProgramsGenerator:
    def __init__(self, adata, save_path, layer='counts_based', encoding_dim=50, n_clusters=250, with_pca=True, pca_components=425, probability_threshold=0.95, epochs=500, batch_size=400, layer_multipliers=[1, 2, 0.75, 0.5, 0.25], device='cpu', lr=0.001, factor=0.1, patience=10, seed=1):
        self.adata = adata
        self.save_path = save_path
        self.layer = layer
        self.encoding_dim = encoding_dim
        self.n_clusters = n_clusters
        self.with_pca = with_pca
        self.pca_components = pca_components
        self.probability_threshold = probability_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.gene_names = None
        self.normalized_matrix = None
        self.encoded_data = None
        self.layer_multipliers = layer_multipliers
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.seed = seed

    def run_pipeline(self):
        self.load_and_prepare_data()
        if self.with_pca:
            self.run_pca()
        self.train_autoencoder_and_encode()
        self.perform_clustering_and_save_results()

    
    def load_and_prepare_data(self):
        matrix = self.adata.layers[self.layer]
        self.gene_names = self.adata.var_names.to_numpy()
        self.normalized_matrix = self.normalize_columns_by_std_csr(matrix).toarray().T

    def run_pca(self):
        pca = PCA(n_components=self.pca_components)
        pca.fit(self.normalized_matrix)
        self.normalized_matrix = pca.transform(self.normalized_matrix)

    def train_autoencoder_and_encode(self):
        self.encoder = self.train_deep_autoencoder(self.normalized_matrix, self.encoding_dim, epochs=self.epochs, batch_size=self.batch_size, device=self.device, lr=self.lr, factor=self.factor, patience=self.patience, layer_multipliers=self.layer_multipliers)
        self.encoded_data = self.encoder(torch.tensor(self.normalized_matrix, dtype=torch.float32).to(self.device)).detach().cpu().numpy()

    def perform_clustering_and_save_results(self):
        distances, _ = self.perform_kmeans_clustering(self.encoded_data, n_clusters=self.n_clusters, seed=self.seed)
        soft_assignments = self.get_soft_assignments_per_cluster_kde(distances)
        cluster_assignments = self.assign_genes_to_clusters_and_sort(soft_assignments, self.gene_names, probability_threshold=self.probability_threshold)
        self.save_cluster_assignments_to_csv(cluster_assignments)

    def train_deep_autoencoder(self, data, encoding_dim, epochs=5000, batch_size=100, device='cpu', lr=0.001, factor=0.1, patience=10, layer_multipliers=[1, 2, 0.75, 0.5, 0.25]):
        device = torch.device(device)
        input_dim = data.shape[1]
        model = Autoencoder(input_dim, encoding_dim, layer_multipliers).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)

        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        dataset = TensorDataset(data_tensor) 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0.0
            for batch_data in dataloader:
                inputs = batch_data[0]
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            epoch_loss /= len(dataloader.dataset)
            scheduler.step(epoch_loss)

            if (epoch+1) % 50 == 0:  
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        
        return model.encoder



    def normalize_columns_by_std_csr(self, sparse_matrix):
        sparse_csc = sparse_matrix.tocsc()
        
        # Calculate the mean for each column (correctly considering zeros)
        column_means = np.array(sparse_csc.mean(axis=0)).ravel()
        
        # Initialize the array for standard deviation
        column_std_dev = np.zeros(sparse_csc.shape[1])
        N = sparse_csc.shape[0]  # Total number of rows
        
        # This loop calculates the variance for each column, taking into account the sparsity of the matrix.
        for i in range(sparse_csc.shape[1]):
            col_data = sparse_csc.data[sparse_csc.indptr[i]:sparse_csc.indptr[i+1]]
            k = len(col_data)  # Number of non-zero elements in the column
            variance = np.sum((col_data - column_means[i])**2) / N + (N-k) * (column_means[i]**2) / N
            column_std_dev[i] = np.sqrt(variance)
        
        # Check for columns with a standard deviation of zero and raise an error
        if column_std_dev[column_std_dev == 0].any():
            raise ValueError("Some genes have a standard deviation of zero. Please remove these columns or handle them separately.")
        
        # Normalize each column by subtracting the mean and dividing by the standard deviation
        for i in range(sparse_csc.shape[1]):
            col_slice = slice(sparse_csc.indptr[i], sparse_csc.indptr[i+1])
            sparse_csc.data[col_slice] = (sparse_csc.data[col_slice] - column_means[i]) / column_std_dev[i]
        
        return sparse_csc.tocsr()


    def get_soft_assignments_per_cluster_kde(self, distances, bandwidth=0.1, eps=EPS, outlier_percentiles=(5, 95), outlier_multiplier=1.5, grid_points=1000):
        n_genes, n_clusters = distances.shape
        soft_assignments = np.zeros((n_genes, n_clusters))
        
        # Invert distances to make closer points have higher values, similar to likelihood
        # Using `EPS` for numerical stability in log transformation
        distances = -np.log(distances + EPS)
        
        for cluster_idx in range(n_clusters):
            features = distances[:, cluster_idx].reshape(-1, 1)
            
            # Calculate Inter-Percentile Range (IPR) and identify outliers
            p_lower, p_upper = np.percentile(features, outlier_percentiles)
            ipr = p_upper - p_lower
            lower_bound = p_lower - outlier_multiplier * ipr
            upper_bound = p_upper + outlier_multiplier * ipr
            
            # Filter out outliers based on the calculated bounds
            non_outlier_indices = np.where((features >= lower_bound) & (features <= upper_bound))[0]
            filtered_features = features[non_outlier_indices]
            
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(filtered_features)
            
            # Generate a grid over the range of distances and evaluate the KDE over this grid
            x_grid = np.linspace(filtered_features.min(), filtered_features.max(), grid_points)
            pdf = np.exp(kde.score_samples(x_grid[:, None]))
            
            # Convert log PDF to CDF by cumulatively summing and normalizing, accounting for grid spacing
            cdf = np.cumsum(pdf) * (x_grid[1] - x_grid[0])
            cdf_interpolation = interp1d(x_grid, cdf, kind='linear', bounds_error=False, fill_value=(0, 1))
            features_cdf = cdf_interpolation(features)
            soft_assignments[:, cluster_idx] = features_cdf.squeeze()
        
        return soft_assignments


    def assign_genes_to_clusters_and_sort(self, soft_assignments, gene_names, probability_threshold=0.95):
        selected_genes_by_cluster = {}
        programs_counter = 0

        for i in range(soft_assignments.shape[1]):
            # Select genes above the threshold
            selected_indices = np.where(soft_assignments[:, i] > probability_threshold)[0]
            selected_genes = gene_names[selected_indices]

            # Proceed if at least 2 genes are selected
            if len(selected_genes) >= 2:
                programs_counter += 1

                gene_soft_assignments = soft_assignments[selected_indices, i]
                # Sort genes by descending soft assignment value
                sorted_indices = np.argsort(-gene_soft_assignments)
                sorted_genes = selected_genes[sorted_indices]
                selected_genes_by_cluster[f'PROGRAM_{programs_counter}'] = sorted_genes.tolist()

        return selected_genes_by_cluster

    def perform_kmeans_clustering(self, data, n_clusters=250, seed=1):

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(data)
        centroids = kmeans.cluster_centers_
        distances = cdist(data, centroids, 'euclidean')
        
        return distances, centroids
    
    def get_soft_assignments_per_cluster_kde(self, distances, bandwidth=0.1, eps=EPS, outlier_percentiles=(5, 95), outlier_multiplier=1.5, grid_points=1000):
        n_genes, n_clusters = distances.shape
        soft_assignments = np.zeros((n_genes, n_clusters))
        
        # Invert distances to make closer points have higher values, similar to likelihood
        # Using `eps` for numerical stability in log transformation
        distances = -np.log(distances + EPS)
        
        for cluster_idx in range(n_clusters):
            features = distances[:, cluster_idx].reshape(-1, 1)
            
            # Calculate Inter-Percentile Range (IPR) and identify outliers
            p_lower, p_upper = np.percentile(features, outlier_percentiles)
            ipr = p_upper - p_lower
            lower_bound = p_lower - outlier_multiplier * ipr
            upper_bound = p_upper + outlier_multiplier * ipr
            
            # Filter out outliers based on the calculated bounds
            non_outlier_indices = np.where((features >= lower_bound) & (features <= upper_bound))[0]
            filtered_features = features[non_outlier_indices]
            
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(filtered_features)
            
            # Generate a grid over the range of distances and evaluate the KDE over this grid
            x_grid = np.linspace(filtered_features.min(), filtered_features.max(), grid_points)
            pdf = np.exp(kde.score_samples(x_grid[:, None]))
            
            # Convert log PDF to CDF by cumulatively summing and normalizing, accounting for grid spacing
            cdf = np.cumsum(pdf) * (x_grid[1] - x_grid[0])
            cdf_interpolation = interp1d(x_grid, cdf, kind='linear', bounds_error=False, fill_value=(0, 1))
            features_cdf = cdf_interpolation(features)
            soft_assignments[:, cluster_idx] = features_cdf.squeeze()
        
        return soft_assignments



    def save_cluster_assignments_to_csv(self, cluster_assignments):
        max_len = max(len(genes) for genes in cluster_assignments.values())
        data_for_df = []
        for program, genes in cluster_assignments.items():
            row = [program] + genes + [''] * (max_len - len(genes))
            data_for_df.append(row)
        df = pd.DataFrame(data_for_df, columns=['Program'] + [f'Gene_{i+1}' for i in range(max_len)])
        df.to_csv(self.save_path, index=False)




class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, layer_multipliers):
        super(Autoencoder, self).__init__()
        
        encoder_layers = []
        current_dim = input_dim
        for multiplier in layer_multipliers:
            next_dim = int(current_dim * multiplier)
            encoder_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(True),
                nn.BatchNorm1d(next_dim)
            ])
            current_dim = next_dim
        encoder_layers.append(nn.Linear(current_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = [nn.Linear(encoding_dim, current_dim)]
        for multiplier in reversed(layer_multipliers):
            next_dim = int(current_dim / multiplier)
            decoder_layers.extend([
                nn.ReLU(True),
                nn.BatchNorm1d(current_dim),
                nn.Linear(current_dim, next_dim),
            ])
            current_dim = next_dim
        decoder_layers.extend([
            nn.ReLU(True),
            nn.BatchNorm1d(current_dim),
            nn.Linear(current_dim, input_dim)
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# %%
