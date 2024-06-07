import torch
import torch.nn as nn
import sys
# from torch_geometric.nn.models import GCN
import torch.nn.functional as F
# from models import MLP
# from models import GCNWithIntermediateOutput, GATWithIntermediateOutput, GraphSAGEWithIntermediateOutput
from utils import create_dummy_adj, add_edges, calc_cosine_sim, calc_degree
import numpy as np
# import copy
from torch_geometric.nn import summary

# from typing import List, Union


class AllGNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            classifier: nn.Module,
            node_generator: nn.Module = None,
            low_degree_updater: nn.Module = None,
            n_add_node: int = 1,
            input_original_degree: bool = False,
            is_input_layer_discriminator_enabled: bool = False,
            is_input_layer_contrastive_enabled: bool = False,
        ):
        super(AllGNN, self).__init__()

        self.classifier = classifier
        self.node_generator = node_generator
        self.low_degree_updater = low_degree_updater
        self.n_add_node = n_add_node
        self.input_original_degree = input_original_degree

        self.is_node_addition_enabled = (self.node_generator is not None)
        self.is_low_degree_updater_enabled = (self.low_degree_updater is not None)
        self.is_discriminator_enabled = is_input_layer_discriminator_enabled
        self.is_contrastive_enabled = is_input_layer_contrastive_enabled
        self.is_fc_in_enabled = self.is_node_addition_enabled \
            or self.is_low_degree_updater_enabled \
            or self.is_discriminator_enabled \
            or self.is_contrastive_enabled

        self.fc_in = nn.Linear(in_channels, hidden_channels)

        self.rng = np.random.default_rng(1234)
    
    def set_hook(self) -> None:
        if hasattr(self.classifier, 'set_hook'):
            self.classifier.set_hook()

    def set_degree(self, low_degree_group: torch.Tensor, degree: torch.tensor):
        self.low_degree_group = low_degree_group
        self.degree = degree
    
    def forward(self, x, adj, **kwargs):
        features = {}
        x = self.fc_in(x)
        # Node Generator
        if self.is_node_addition_enabled:
            # TODO: adjをcsr形式で処理する
            dummy_adj, cut_nodes = create_dummy_adj(adj, self.degree, self.rng)
            generated_nodes = self.node_generator(x, dummy_adj)
            low_degree_idx = torch.where(cut_nodes==-1)[0]
            non_low_degree_idx = torch.where(cut_nodes!=-1)[0]
            # n_add = len(low_degree_idx)
            features['truth_emb'] = x[cut_nodes[non_low_degree_idx]]
            features['predicted_emb'] = generated_nodes[non_low_degree_idx]
            x, adj = add_edges(x, adj, low_degree_idx, generated_nodes)
            for _ in range(self.n_add_node - 1):
                generated_nodes = self.node_generator(x, adj)
                x, adj = add_edges(x, adj, low_degree_idx, generated_nodes)
            # new_edge_weights = self.generated_node_weight * torch.ones(n_add).cuda()
            # edge_weights = torch.concatenate([edge_weights, new_edge_weights], dim=0)
        
        # Low degree Updater
        features['contrastive_target'] = [None, None]
        features['regularization_target'] = torch.zeros_like(x)
        if self.is_low_degree_updater_enabled:
            ldu_output = self.low_degree_updater(x, adj)
            features['regularization_target'] = features['regularization_target'] + ldu_output
            x = x + ldu_output
        features['contrastive_target'][0] = x
        features['discr_target'] = x
        
        # Main GNN
        if self.input_original_degree:
            degree = kwargs['d']
        else:
            degree = calc_degree(adj)
            degree = degree.unsqueeze(-1)
        classifier_kwargs = {
            'd': degree,
            'idx': kwargs['idx'],
            'edge': kwargs['edge'],
            'head': kwargs['head'],
        }
        x, classifier_features = self.classifier(x, adj, **classifier_kwargs)
        features['contrastive_target'][1] = classifier_features['contrastive_target']
        if 'b' in classifier_features:
            features['b'] = classifier_features['b']
        if 'film' in classifier_features:
            features['film'] = classifier_features['film']
        if 'relation_output' in classifier_features:
            features['relation_output'] = classifier_features['relation_output']
        
        return x, features


class LinkPredictor(nn.Module):
    def __init__(
            self,
            base_model: nn.Module,
            in_channels: int = None,
            hidden_channels: int = None,
            num_layers: int = None,
            dropout: int = None,
            similarity_based = True,
        ):
        super(LinkPredictor, self).__init__()

        if not similarity_based:
            if in_channels is None or hidden_channels is None or num_layers is None or dropout is None:
                print("If similarity_based is False, in_channels, hidden_channels, num_layers, dropout must be specified", file=sys.stderr)
                exit(1)

        self.base_model = base_model
        self.similarity_based = similarity_based

        if not similarity_based:
            # TODO: Implement this
            print("Not implemented yet", file=sys.stderr)
            exit(0)

    def forward(self, x, adj, srcs, drts, all_pair: bool = False):
        if self.similarity_based:
            x = self.base_model(x, adj)
            if all_pair:
                output = calc_cosine_sim(x[srcs], x[drts])
            else:
                output = F.cosine_similarity(x[srcs], x[drts])
            output = (output + 1.) / 2.
            return output
        else:
            # TODO: implement this
            return None
