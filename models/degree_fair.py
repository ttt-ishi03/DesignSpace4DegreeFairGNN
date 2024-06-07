import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.sparse as sp
# import torch.optim as optim
from layers import Debias_v2, Debias_v3, Debias_v4
import numpy as np



def convert_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class DFairGNN_2(nn.Module):
    def __init__(self, args, nfeat, nclass, max_degree, sam):
        super(DFairGNN_2, self).__init__()
        self.dropout = args.dropout
        self.debias1 = Debias_v3(args, nfeat, args.dim, max_degree, sam)
        self.debias2 = Debias_v3(args, args.dim, args.dim, max_degree, sam)
        self.fc = nn.Linear(args.dim, nclass)


    def forward(self, x, adj, **kwargs):
        d = kwargs['d']
        idx = kwargs['idx']
        features = {}
        x, features1 = self.debias1(x, adj, d, idx)
        b1 = features1['L_b']
        film1 = features['L_film']
        x = F.leaky_relu(x)
        features['contrastive_target'] = x
        x = F.dropout(x, self.dropout, training=self.training)

        x, features2 = self.debias2(x, adj, d, idx)
        b2 = features2['L_b']
        film2 = features2['L_film']
        x = F.leaky_relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)

        features['b'] = b1+b2
        features['film'] = film1+film2
        return x, features


class DFair_GCN(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            dropout,
            base,
            dataset,
            dim_d,
            omega,
            k,
            max_degree,
            no_structural_contrast=False,
            no_modulation=False,
            random_miss=False,
            no_miss=False,
            no_localization=False,
        ):
        super(DFair_GCN, self).__init__()
        self.dropout = dropout
        self.debias1 = Debias_v4(
            in_channels,
            hidden_channels,
            base,
            dataset,
            dim_d,
            omega,
            k,
            max_degree,
            no_structural_contrast=no_structural_contrast,
            no_modulation=no_modulation,
            random_miss=random_miss,
            no_miss=no_miss,
            no_localization=no_localization,
        )
        self.debias2 = Debias_v4(
            hidden_channels,
            hidden_channels,
            base,
            dataset,
            dim_d,
            omega,
            k,
            max_degree,
            no_structural_contrast=no_structural_contrast,
            no_modulation=no_modulation,
            random_miss=random_miss,
            no_miss=no_miss,
            no_localization=no_localization,
        )
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.no_structural_contrast = no_structural_contrast
        self.no_modulation = no_modulation

    def set_hook(self):
        if hasattr(self.debias1, 'set_hook'):
            self.debias1.set_hook()
        if hasattr(self.debias2, 'set_hook'):
            self.debias2.set_hook()

    def forward(self, x, adj, **kwargs):
        d = kwargs['d']
        idx = kwargs['idx']
        edge = kwargs['edge']
        head = kwargs['head']
        features = {}
        x, features1 = self.debias1(x, adj, d, idx, edge, head)
        b1 = features1['L_b']
        film1 = features1['L_film']
        x = F.leaky_relu(x)
        features['contrastive_target'] = x
        x = F.dropout(x, self.dropout, training=self.training)

        x, features2 = self.debias2(x, adj, d, idx, edge, head)
        b2 = features2['L_b']
        film2 = features2['L_film']
        x = F.leaky_relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)

        features['b'] = b1+b2
        features['film'] = film1+film2
        if 'relation_output' in features1.keys() and 'relation_output' in features2.keys():
            features['relation_output'] = [features1['relation_output'], features2['relation_output']]
        return x, features


class DFair_GAT(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            dropout,
            base,
            dataset,
            dim_d,
            omega,
            k,
            max_degree,
            nheads=3,
            no_structural_contrast=False,
            no_modulation=False,
            random_miss=False,
            no_miss=False,
            no_localization=False,
        ):
        super(DFair_GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            Debias_v4(
                in_channels,
                hidden_channels,
                base,
                dataset,
                dim_d,
                omega,
                k,
                max_degree,
                no_structural_contrast=no_structural_contrast,
                no_modulation=no_modulation,
                random_miss=random_miss,
                no_miss=no_miss,
                no_localization=no_localization,
            ) for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = Debias_v4(
            hidden_channels*nheads,
            out_channels,
            base,
            dataset,
            dim_d,
            omega,
            k,
            max_degree,
            no_structural_contrast=no_structural_contrast,
            no_modulation=no_modulation,
            random_miss=random_miss,
            no_miss=no_miss,
            no_localization=no_localization,
        )


    def forward(self, x, adj, **kwargs):
        d = kwargs['d']
        idx = kwargs['idx']
        edge = kwargs['edge']
        head = kwargs['head']
        features = {}
        x = F.dropout(x, self.dropout, training=self.training)
        
        x1 = []
        b1 = 0
        film1 = 0
        relation_output = []
        for att in self.attentions:
            at_x1, at_features1 = att(x, adj, d, idx, edge, head)
            at_b1 = at_features1['L_b']
            at_film1 = at_features1['L_film']
            at_x1 = F.leaky_relu(at_x1)
            x1.append(at_x1)
            b1 += at_b1
            film1 += at_film1
            if 'relation_output' in at_features1.keys():
                relation_output.append(at_features1['relation_output'])

        x = torch.cat(x1, dim=1)
        features['contrastive_target'] = x
        b1 /= len(self.attentions)
        film1 /= len(self.attentions)

        x = F.dropout(x, self.dropout, training=self.training)
        x, at_features2 = self.out_att(x, adj, d, idx, edge, head)
        b2 = at_features2['L_b']
        film2 = at_features2['L_film']

        features['b'] = b1+b2
        features['film'] = film1+film2
        if len(relation_output) > 0  and 'relation_output' in at_features2.keys():
            features['relation_output'] = relation_output + [at_features2['relation_output']]
        return x, features


class DFair_Sage(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            dropout,
            base,
            dataset,
            dim_d,
            omega,
            k,
            max_degree,
            no_structural_contrast=False,
            no_modulation=False,
            random_miss=False,
            no_miss=False,
            no_localization=False,
        ):
        super(DFair_Sage, self).__init__()
        if isinstance(hidden_channels, int):
            nhid1 = nhid2 = hidden_channels
        else:
            nhid1, nhid2 = hidden_channels
        self.dropout = dropout
        self.debias1 = Debias_v4(
            in_channels,
            nhid1,
            base,
            dataset,
            dim_d,
            omega,
            k,
            max_degree,
            no_structural_contrast=no_structural_contrast,
            no_modulation=no_modulation,
            random_miss=random_miss,
            no_miss=no_miss,
            no_localization=no_localization,
        )
        self.debias2 = Debias_v4(
            nhid1,
            nhid2,
            base,
            dataset,
            dim_d,
            omega,
            k,
            max_degree,
            no_structural_contrast=no_structural_contrast,
            no_modulation=no_modulation,
            random_miss=random_miss,
            no_miss=no_miss,
            no_localization=no_localization,
        )
        self.fc = nn.Linear(nhid2, out_channels)

    def set_hook(self):
        if hasattr(self.debias1, 'set_hook'):
            self.debias1.set_hook()
        if hasattr(self.debias2, 'set_hook'):
            self.debias2.set_hook()
    
    def forward(self, x, adj, **kwargs):
        d = kwargs['d']
        idx = kwargs['idx']
        edge = kwargs['edge']
        head = kwargs['head']
        features_ret = {}
        #x = F.dropout(x, self.dropout, training=self.training)
        x, features1 = self.debias1(x, adj, d, idx, edge, head)
        x = F.leaky_relu(x)
        b1 = features1['L_b']
        film1 = features1['L_film']
        features_ret['contrastive_target'] = x
        x = F.dropout(x, self.dropout, training=self.training)

        x, features2 = self.debias2(x, adj, d, idx, edge, head)
        x = F.leaky_relu(x)
        b2 = features2['L_b']
        film2 = features2['L_film']
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)

        features_ret['b'] = b1+b2
        features_ret['film'] = film1+film2

        return x, features_ret
