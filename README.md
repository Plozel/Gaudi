# Gaudi - Computational Framework for Analyzing Spatial Transcriptomics Data Through Community-Level Perspective

This repository contains a computational framework, named Gaudi, I developed during my MSc thesis at the Technion, under the supervision of Dr. Dvir Aran.
The framework is designed for analyzing spatial transcriptomics data at the cellular community level. 
It leverages machine learning models like graph neural networks, statistical tests, and self-supervised learning to provide insights into group-level phenomena such as cellular interactions and community dynamics.


## Table of Contents
- [Installation](#installation)
- [Additional Resources](#additional-resources)
- [Usage](#usage)
- [Features](#features)
- [Examples](#examples)


## Installation
To install this framework, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Plozel/Gaudi.git
cd Gaudi

conda env create -f environment.yml
conda activate gaudi_env

python setup.py build
python setup.py install --user
```


## Additional Resources

To run some of the examples in this repository, you'll need to download additional files. These files are available at the following link:

[Download Additional Files](https://drive.google.com/drive/folders/1zo4JjzSbUj5nFiKJxB_EiXIM8MOXsKFT?usp=sharing)

Please make sure to place these files in the appropriate directory as specified in the individual example notebooks.


## Usage
Here is the most basic example of how to use the framework:

```python
import gaudi
import scanpy as sc

# Read AnnData object from the specified file path
adata = sc.read_h5ad("path to your anndata.h5ad")

# Instantiate GaudiObject with configurations
gaudi_obj = gaudi.core.GaudiObject(adata)

# Generate representations using Gaudi's learning procedure, enabling training mode
gaudi_obj.generate_representations(train=True)

# Generate and plot community-level labels using Leiden clustering 
# (Defaults to using Gaudi's learned representations but can use any other specified group of features)
labels_name = "gaudi_labels"
level = 1 # Set to 0 for cell level and 1 for community level
gaudi_obj.generate_labels(level=level, labels_name=labels_name, clustering_method='leiden', n_neighbors=250, resolution=0.5)
gaudi_obj.plot_communities(color_by=labels_name)

# Perform statistical comparison to identify highly differentiable genes between communities based on their mean gene expression
data_type = 'mean' # This can be any key found in GaudiObject.perspectives
layer = 'log_normalized_based'
significant_genes_level_1, significant_genes_dict_level_1 = gaudi_obj.compare(level=level, groupby=labels_name, data_type=data_type)
gaudi_obj.plot_dotplot(data_type=data_type, level=level, var_names=significant_genes_dict_level_1, layer=layer, groupby=labels_name)

# Calculate and plot the distribution of cell types within the community classes
# Note: Cell type information should be stored in GaudiObject.sample.labels[0]['celltype_key']
gaudi_obj.plot_composition('gaudi_labels', 'celltype_key', 1, 0)

# Perform statistical comparison based on cell type composition within communities
data_type = 'composition'
significant_celltypes_level_1, significant_celltypes_dict_level_1 = gaudi_obj.compare(level=level, groupby=labels_name, data_type=data_type)
gaudi_obj.plot_dotplot(data_type=data_type, level=level, var_names=significant_celltypes_dict_level_1, groupby=labels_name)
```


## Features
- Detection of Spatially-dense communities
- Learning procedure for community-level classification and tissue segmentation
- Interpretable community-level features
- Tools for Statistical comparison
- Visualization tools for easy interpretation of results
- Easy to use automatic experiments management


## Examples
You can find example Jupyter notebooks demonstrating various use cases in the `examples` directory:

- [Mouse Olfactory Bulb (StereoSeq)](examples/mouse_olfactory_bulb/mouse_olfactory_bulb.ipynb)
- [Mouse Hippocampus (SlideSeq)](examples/mouse_hippocampus/mouse_hippocampus.ipynb)
- [Human Lymph Node (Xenium)](examples/lymph_node/lymph_node.ipynb)

