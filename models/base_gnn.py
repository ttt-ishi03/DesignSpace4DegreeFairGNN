from torch_geometric.nn.models import GCN, GAT, GraphSAGE

from typing import List, Optional

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor
import torch
import torch.nn as nn


class GCNWithIntermediateOutput(GCN):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout, intermediate_layer=0, **kwargs):
        super(GCNWithIntermediateOutput, self).__init__(in_channels, hidden_channels, num_layers, out_channels, dropout, **kwargs)
        self.intermediate_layer = intermediate_layer
        self._need_low_degree_group = False
    
    def need_low_degree_group(self):
        return self._need_low_degree_group

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
        **kwargs,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        """
        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs: List[Tensor] = []
        for i in range(self.num_layers):
            if num_sampled_nodes_per_hop is not None:
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight,
                                  edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            if i == self.intermediate_layer:
                intermediate_output = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        features = {'contrastive_target': intermediate_output}
        return x, features


class GATWithIntermediateOutput(GAT):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout, intermediate_layer=0, **kwargs):
        super(GATWithIntermediateOutput, self).__init__(in_channels, hidden_channels, num_layers, out_channels, dropout, **kwargs)
        self.intermediate_layer = intermediate_layer
        self._need_low_degree_group = False
    
    def need_low_degree_group(self):
        return self._need_low_degree_group
    
    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
        **kwargs,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        """
        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs: List[Tensor] = []
        for i in range(self.num_layers):
            if num_sampled_nodes_per_hop is not None:
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight,
                                  edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            if i == self.intermediate_layer:
                intermediate_output = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        features = {'contrastive_target': intermediate_output}
        return x, features

class GraphSAGEWithIntermediateOutput(GraphSAGE):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout, intermediate_layer=0, **kwargs):
        super(GraphSAGEWithIntermediateOutput, self).__init__(in_channels, hidden_channels, num_layers, out_channels, dropout, **kwargs)
        self.intermediate_layer = intermediate_layer
        self._need_low_degree_group = False
    
    def need_low_degree_group(self):
        return self._need_low_degree_group

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
        **kwargs,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        """
        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs: List[Tensor] = []
        for i in range(self.num_layers):
            if num_sampled_nodes_per_hop is not None:
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight,
                                  edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            if i == self.intermediate_layer:
                intermediate_output = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        features = {'contrastive_target': intermediate_output}
        return x, features


class MLP(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_layers,
            out_channels,
            dropout=0.5,
            is_dropout_enabled=True,
            is_batchnorm_enabled=True,
        ):
        super(MLP, self).__init__()

        self.input_layer = nn.Linear(in_channels, hidden_channels)

        middle_layers = []
        for _ in range(num_layers - 2):
            middle_layers.append(nn.Linear(hidden_channels, hidden_channels))
            if is_batchnorm_enabled:
                middle_layers.append(nn.BatchNorm1d(hidden_channels))
            middle_layers.append(nn.ReLU())
            if is_dropout_enabled:
                middle_layers.append(nn.Dropout(dropout))

        self.middle_layers = nn.Sequential(*middle_layers)

        self.output_layer = nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, x, _):
        x = self.input_layer(x)
        x = self.middle_layers(x)
        x = self.output_layer(x)

        return x


class RandomModel(nn.Module):
    def __init__(self, nclass):
        super(RandomModel, self).__init__()
        self.nclass = nclass

    def __call__(self, x, _):
        return torch.rand(x.shape[0], self.nclass)