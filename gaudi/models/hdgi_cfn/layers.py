import torch

class CommunityFocusedNetwork(torch.nn.Module):
    """Encodes node and community information using specified graph convolution layers with customizable depth."""

    def __init__(self, input_dim, hidden_dim, output_dim, 
                 conv_layer, num_node_layers, num_community_layers):
        super(CommunityFocusedNetwork, self).__init__()

        # Validate the minimum number of layers
        assert num_node_layers >= 1, "There should be at least one node layer."
        assert num_community_layers >= 1, "There should be at least one community layer."
        
        # Create node-to-node connection layers
        self.node_convs = torch.nn.ModuleList()
        self.node_convs.append(conv_layer(input_dim, hidden_dim))
        for _ in range(num_node_layers - 1):
            self.node_convs.append(conv_layer(hidden_dim, hidden_dim))
        
        # Create node-to-community connection layers
        self.community_convs = torch.nn.ModuleList()
        for _ in range(num_community_layers - 1):
            self.community_convs.append(conv_layer(hidden_dim, hidden_dim))
        self.community_convs.append(conv_layer(hidden_dim, output_dim))
        
        # Activation layers for all convolutional layers
        self.activations = torch.nn.ModuleList([torch.nn.PReLU(hidden_dim) for _ in range(num_node_layers + num_community_layers - 1)])
        self.activations.append(torch.nn.PReLU(output_dim))

    def forward(self, x, edge_index, community_edge_index):
        # Process node-to-node connections
        for conv, activation in zip(self.node_convs, self.activations[:len(self.node_convs)]):
            x = activation(conv(x, edge_index))

        # Process node-to-community connections
        for conv, activation in zip(self.community_convs, self.activations[len(self.node_convs):]):
            x = activation(conv(x, community_edge_index))

        return x



def corruption(x, edge_index, community_edge_index, *args, **kwargs):
    """Shuffles the node features for corruption."""

    shuffled_x = x[torch.randperm(x.size(0)), :]
    return shuffled_x, edge_index, community_edge_index


def summary(node_x, community_x, *args, **kwargs):
    """Computes the summary for nodes and communities."""

    node_summary = node_x.mean(dim=0)
    community_summary = community_x.mean(dim=0)

    return node_summary, community_summary
