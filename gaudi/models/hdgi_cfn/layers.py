import torch
import torch.nn as nn
import torch.nn.functional as F



class CommunityFocusedNetwork(nn.Module):
    """
    Encodes node and community information in a hierarchical manner by interleaving
    node-to-node and node-to-community graph convolutional layers. Supports customizable
    depth, convolutional types, and activation functions for capturing both local and 
    global graph interactions.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, conv_layer, activation_fn, num_layers):
        super(CommunityFocusedNetwork, self).__init__()
        
        assert num_layers >= 1, "The model must have at least one layer."
        self.num_layers = num_layers

        # Create interleaved node-to-node and node-to-community layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            node_layer = conv_layer(input_dim if i == 0 else hidden_dim, hidden_dim)
            community_layer = conv_layer(hidden_dim, hidden_dim if i < self.num_layers - 1 else output_dim)
            self.layers.append(nn.Sequential(node_layer, activation_fn(), community_layer, activation_fn()))
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            for module in layer:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

    def forward(self, x, edge_index, community_edge_index):
        for layer in self.layers:
            x = layer[1](layer[0](x, edge_index))
            x = layer[3](layer[2](x, community_edge_index))
        return x
        
def corruption(x, edge_index, community_edge_index, *args, **kwargs):
    """
    Corrupts the input graph by shuffling the adjacency indices.
    """

    # Shuffle the adjacency indices along the first dimension
    shuffled_edge_index = edge_index.clone()
    shuffled_edge_index[0] = shuffled_edge_index[0, torch.randperm(edge_index.size(1))]

    # Shuffle the community adjacency indices along both dimensions
    shuffled_community_edge_index = community_edge_index.clone()
    shuffled_community_edge_index = community_edge_index[:, torch.randperm(community_edge_index.size(1))]

    return x, shuffled_edge_index, shuffled_community_edge_index

def summary(node_x, community_x, *args, **kwargs):
    """
    Computes node and community summaries by applying sigmoid activation and averaging.
    """

    node_summary = torch.sigmoid(node_x).mean(dim=0)
    community_summary = torch.sigmoid(community_x).mean(dim=0)
    return node_summary, community_summary

