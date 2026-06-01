"""
Graph Neural Network Encoder
This module implements a Graph Neural Network (GNN) encoder using PyTorch and PyTorch Geometric.
It builds a k-nearest neighbour (k-NN) graph from node features, applies an EdgeConv layer,
and then pools the node embeddings to produce a fixed-size representation for each graph.
It naturally handles different numbers of nodes per example.
"""

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool


def collate_fn_gnn(batch: list[dict]) -> tuple[Batch, torch.Tensor]:
    """
    Custom function that defines how batches are formed.

    For a more complicated dataset with variable length per event and Graph Neural Networks,
    we need to define a custom collate function which is passed to the DataLoader.
    The default collate function in PyTorch Geometric is not suitable for this case.

    This function takes the Awkward arrays, converts them to PyTorch tensors,
    and then creates a PyTorch Geometric Data object for each event in the batch.

    You do not need to change this function.

    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list = []
    labels = []

    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the features are (time, x, y)
        # for training a GNN, we need the graph notes, i.e., the individual hits, as the first dimension,
        # so we need to transpose to get (n_hits, n_features)
        tensor_data = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensor_data = tensor_data.to(dtype=torch.float32)

        # PyTorch Geometric needs the data in a specific format
        # we need to create a PyTorch Geometric Data object for each event
        this_graph_item = Data(x=tensor_data)
        data_list.append(this_graph_item)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    labels = torch.cat(labels, dim=0)  # convert the list of tensors to a single tensor
    packed_data = Batch.from_data_list(data_list)  # convert the list of Data objects to a single Batch object
    return packed_data, labels


# the MLP is defined as a simple two-layer MLP with ReLU activation
def MLP(in_channels: int, hidden_channels: int, out_channels: int) -> nn.Sequential:
    """
    A simple multi-layer perceptron (MLP) with two linear layers and ReLU activation.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Number of hidden features.
    out_channels : int
        Number of output features.
    Returns
    -------
    torch.nn.Sequential
        A sequential model containing two linear layers with ReLU activation.
    """
    return nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder that builds a k-NN graph from node features,
    applies an EdgeConv layer, and then pools the node embeddings to produce
    a fixed-size representation for each graph.

    It naturally handles different numbers of nodes per example.
    """

    def __init__(self, input_dim: int, graph_intermediate_dim: int, output_dim: int, k_nearest_neighbour: int = 3, num_layers: int = 3) -> None:
        """
        Definition of the Graph Neural Network.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        graph_intermediate_dim : int
            Number of intermediate features for the GNN layers.
        output_dim : int
            Number of output features.
        k_nearest_neighbour : int
            Number of nearest neighbors to consider for the k-NN graph.
        num_layers : int
            Number of GNN layers to apply.
        """
        super(GNNEncoder, self).__init__()

        # the hidden dimension of the MLPs used in the EdgeConv layers
        self.hidden_mlp_dim = 128

        # number of GNN layers
        self.num_layers = num_layers
        assert self.num_layers > 0

        # the dimension of the graph layers
        # for the first layer, it is the input dimension (i.e. time, x and y position)
        # for the other layers, it is the graph_intermediate_dim
        self.graph_dims = [input_dim]
        self.graph_dims.extend([graph_intermediate_dim] * self.num_layers)

        self.k = k_nearest_neighbour

        # we save each network layer in a list
        self.layer_list = torch.nn.ModuleList()
        for current_layer in range(len(self.graph_dims) - 1):
            # the EdgeConv layer is a dynamic layer, i.e. it computes the k-NN graph on the fly and applies a neural network (typically an MLP) to each pair of nodes
            # the input dimension is 2 * input_dimension, because DynamicEdgeConv concatenates the features of the two nodes that are connected by the edge
            self.layer_list.append(
                DynamicEdgeConv(MLP(2 * self.graph_dims[current_layer], self.hidden_mlp_dim, self.graph_dims[current_layer + 1]), aggr="mean", k=self.k)
            )

        # the final MLP layer that maps the output of the last GNN layer to the output dimension
        self.final_mlp = MLP(self.graph_dims[-1], self.hidden_mlp_dim, output_dim)
        return

    def forward(self, data: Any) -> nn.Sequential:
        # data is a batch graph item. it contains a list of tensors (x) and how the batch is structured along this list (batch)
        x = data.x
        batch = data.batch

        # go through individual dynamic edge convolution layers - the knn graph is recomputed every time automatically by the
        # DynamicEdgeConv layer
        for layer in self.layer_list:
            x = layer(x, batch)

        # the output of the last layer has dimensions (n_batch, n_nodes, graph_intermediate_dim)
        # where n_batch is the number of graphs in the batch and n_nodes is the number of nodes in the graph
        # i.e. one output per node (i.e. the hits in the event).
        # To combine all node features into single prediction, we pool the node features
        x = global_mean_pool(x, batch)  # -> (n_batch, output_dim)
        # x is now a tensor of shape (n_batch, output_dim)

        # final mlp mapping to map to the output dimension to the number of labels we want to predict
        x = self.final_mlp(x)

        return x
