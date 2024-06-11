import sys

import torch
import torch.nn as nn
# from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
# import math
# import scipy.sparse as sp
import numpy as np
from .relation import Relation
from .relation_v2 import Relationv2
from .generators import Generator
from .gat import SpGraphAttentionLayer

EPS = 1e-5


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert not indices.requires_grad
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class Debias_v2(nn.Module):
    """
    original DegFairGNN code
    """
    def __init__(self, in_channels, out_channels, base, dataset, dim_d, omega, k, d_max):
        super(Debias_v2, self).__init__()

        self.dim_M = dim_d
        self.out_channels = out_channels
        self.omega = omega
        self.d_max = (d_max+512) #0->dmax
        self.base = base
        self.dataset = dataset
        self.k = k
        #self.w = args.w
        #self.sparse = args.sparse

        
        self.weight = nn.Linear(in_channels, out_channels)
        if base == 2:
            self.a = nn.Parameter(torch.zeros(size=(1, 2*out_channels)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)
            self.special_spmm = SpecialSpmm()
            self.dropout = nn.Dropout()

        self.W_gamma = nn.Parameter(torch.FloatTensor(self.dim_M, out_channels))
        self.W_beta = nn.Parameter(torch.FloatTensor(self.dim_M, out_channels))
        self.U_gamma = nn.Parameter(torch.FloatTensor(out_channels, out_channels))
        self.U_beta = nn.Parameter(torch.FloatTensor(out_channels, out_channels))
        self.b_gamma = nn.Parameter(torch.FloatTensor(1, out_channels))
        self.b_beta = nn.Parameter(torch.FloatTensor(1, out_channels))

        self.W_add = nn.Linear(out_channels, out_channels, bias=False)
        self.W_rev = nn.Linear(out_channels, out_channels, bias=False)

        # Positional Encoding
        PE = np.array([
            [pos / np.power(10000, (i-i%2)/self.dim_M) for i in range(self.dim_M)]
            for pos in range(self.d_max)])

        PE[:, 0::2] = np.sin(PE[:, 0::2]) 
        PE[:, 1::2] = np.cos(PE[:, 1::2]) 
        self.PE = torch.as_tensor(PE, dtype=torch.float32)
        self.PE = self.PE.cuda()
        self.set_parameters()


    def set_parameters(self):
        #nn.init.uniform_(self.m)
        nn.init.uniform_(self.W_gamma)
        nn.init.uniform_(self.W_beta)
        nn.init.uniform_(self.U_gamma)
        nn.init.uniform_(self.U_beta)
        nn.init.uniform_(self.b_gamma)
        nn.init.uniform_(self.b_beta)

        '''
            M_stdv = 1. / math.sqrt(self.M.size(1))
            self.M.data.uniform_(-M_stdv, M_stdv)

            b_stdv = 1. / math.sqrt(self.b.size(1))
            self.b.data.uniform_(-b_stdv, b_stdv)

            for m in self.modules():
                print(m.weight)
        '''

    def forward(self, x, adj, degree, idx, edge):
        h = self.weight(x)
        m_dv = torch.squeeze(self.PE[degree])
        m_dv = m_dv.cuda()

        # version 1
        if self.dataset != 'nba':
            h *= self.dim_M**0.5
        gamma = F.leaky_relu(torch.matmul((m_dv), self.W_gamma) + self.b_gamma) #
        beta = F.leaky_relu(torch.matmul((m_dv), self.W_beta) + self.b_beta) #
        

        #neighbor mean
        i = torch.spmm(adj, h)
        i = i / degree
        i[torch.where(degree==0)[0]] = 0.    
        assert not torch.isnan(i).any()

        # debias low-degree
        b_add = (gamma + 1) * self.W_add(i) + beta
        #b_add = self.W_add(i)

        # debias high-degree
        b_rev = (gamma + 1) * self.W_rev(i) + beta
        #b_rev = self.W_rev(i)

        mean_degree = torch.mean(degree.float())
        K = mean_degree * self.k
        R = torch.where(degree < K, torch.cuda.FloatTensor([1.]), torch.cuda.FloatTensor([0.]))


        #b_rev = b_add
        # compute constraints
        L_b = torch.sum(torch.norm((R*b_add)[idx], dim=1)) + torch.sum(torch.norm(((1-R)*b_rev)[idx], dim=1))
        L_b /= idx.shape[0]

        L_film = torch.sum(torch.norm(gamma[idx], dim=1)) + torch.sum(torch.norm(beta[idx], dim=1))
        L_film /= idx.shape[0]

        bias = self.omega * (R * b_add - (1-R) * b_rev)
        #bias = self.omega * b_add
        #bias = 0

        if self.base == 1:
            output = torch.mm(adj, h) + h + bias
            output /= (degree + 1)

        elif self.base == 2:

            dv = 'cuda' if x.is_cuda else 'cpu'
            N = x.size()[0]
            
            # h: N x out
            #assert not torch.isnan(h).any()

            # Self-attention on the nodes - Shared attention mechanism
            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
            # edge: 2*D x E

            edge_e = torch.exp(-F.leaky_relu(self.a.mm(edge_h).squeeze())) #, negative_slope=0.2))
            assert not torch.isnan(edge_e).any()
            # edge_e: E

            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
            # e_rowsum: N x 1

            edge_e = self.dropout(edge_e)
            # edge_e: E
            
            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            assert not torch.isnan(edge).any()
            torch.set_printoptions(edgeitems=10000)
            # print(h_prime[0])
            assert not torch.isnan(h_prime).any()
            
            # h_prime: N x out    
            h_prime = h_prime + bias
            output = h_prime.div((e_rowsum+1e-5)+1)

            # h_prime: N x out
            assert not torch.isnan(output).any()
        
        elif self.base == 3:
            neighbor = torch.spmm(adj, x)
            ft_neighbor = self.weight(neighbor)
            ft_neighbor += bias 
            ft_neighbor /= (degree + 1)

            output = torch.cat([h, ft_neighbor], dim=1)

        features = {'L_b': L_b, 'L_film': L_film}
        return output, features


class Debias_v3(nn.Module):
    """
    modified DegFairGNN code
    """
    def __init__(self, in_channels, out_channels, base, dataset, dim_d, omega, k, d_max, dropout=0.5, nheads=1, no_structural_contrast=False, no_modulation=False):
        super(Debias_v3, self).__init__()

        self.dim_M = dim_d
        self.out_channels = out_channels
        self.omega = omega
        self.d_max = (d_max+512) #0->dmax
        self.base = base
        self.dataset = dataset
        self.k = k
        self.dropout = dropout
        self.nheads = nheads
        self.no_structural_contrast = no_structural_contrast
        self.no_modulation = no_modulation
        #self.w = args.w
        #self.sparse = args.sparse

        # self.weight = nn.Linear(in_channels, out_channels)
        if base == 1:
            self.nheads = 1
            self.conv = GCNConv(
                in_channels,
                out_channels,
                improved=False,
                cached=False,
                add_self_loops=True,
                normalize=True,
                bias=False,
            )
        elif base == 2:
            self.conv = GATConv(
                in_channels,
                out_channels,
                heads=self.nheads,
                concat=True,
                negative_slope=0.01,
                dropout=self.dropout,
                add_self_loops=True,
                edge_dim=None,
                bias=False,
            )
            # self.a = nn.Parameter(torch.zeros(size=(1, 2*out_channels)))
            # nn.init.xavier_uniform_(self.a.data, gain=1.414)
            # self.special_spmm = SpecialSpmm()
            # self.dropout = nn.Dropout()
        elif base == 3:
            self.nheads = 1
            self.conv = SAGEConv(
                in_channels,
                out_channels,
                aggr = 'mean',
                normalize=False,
                root_weight=False,
                project=False,
                bias=False,
            )
            self.conv.lin_l.register_forward_hook(self.hook)

        self.set_hook()

        self.W_gamma = nn.Parameter(torch.FloatTensor(self.dim_M, out_channels))
        self.W_beta = nn.Parameter(torch.FloatTensor(self.dim_M, out_channels))
        self.U_gamma = nn.Parameter(torch.FloatTensor(out_channels, out_channels))
        self.U_beta = nn.Parameter(torch.FloatTensor(out_channels, out_channels))
        self.b_gamma = nn.Parameter(torch.FloatTensor(1, out_channels))
        self.b_beta = nn.Parameter(torch.FloatTensor(1, out_channels))

        self.W_add = nn.Linear(out_channels, self.nheads * out_channels, bias=False)  # nheads was modified to 1 when base isn't 2
        self.W_rev = nn.Linear(out_channels, self.nheads * out_channels, bias=False)

        # Positional Encoding
        PE = np.array([
            [pos / np.power(10000, (i-i%2)/self.dim_M) for i in range(self.dim_M)]
            for pos in range(self.d_max)])

        PE[:, 0::2] = np.sin(PE[:, 0::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])
        self.PE = torch.as_tensor(PE, dtype=torch.float32)
        self.PE = self.PE.cuda()
        self.set_parameters()

    def set_hook(self):
        if self.base == 1:
            self.conv.lin.register_forward_hook(self.hook)
        elif self.base == 2:
            self.conv.lin_src.register_forward_hook(self.hook)
        elif self.base == 3:
            self.conv.lin_l.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.lin_feat = output

    def set_parameters(self):
        #nn.init.uniform_(self.m)
        nn.init.uniform_(self.W_gamma)
        nn.init.uniform_(self.W_beta)
        nn.init.uniform_(self.U_gamma)
        nn.init.uniform_(self.U_beta)
        nn.init.uniform_(self.b_gamma)
        nn.init.uniform_(self.b_beta)

        '''
            M_stdv = 1. / math.sqrt(self.M.size(1))
            self.M.data.uniform_(-M_stdv, M_stdv)

            b_stdv = 1. / math.sqrt(self.b.size(1))
            self.b.data.uniform_(-b_stdv, b_stdv)

            for m in self.modules():
                print(m.weight)
        '''

    def forward(self, x, adj, degree, idx, edge, **kwargs):
        output = self.conv(x, adj)
        h = self.lin_feat  # self.lin_feat is the intermediate feature of conv layer. It is stored in hook function

        m_dv = torch.squeeze(self.PE[degree])
        m_dv = m_dv.cuda()

        # version 1
        if self.dataset != 'nba':
            h *= self.dim_M**0.5
        gamma = F.leaky_relu(torch.matmul((m_dv), self.W_gamma) + self.b_gamma) #
        beta = F.leaky_relu(torch.matmul((m_dv), self.W_beta) + self.b_beta) #

        #neighbor mean
        i = torch.spmm(adj, h)
        i = i / degree
        i[torch.where(degree==0)[0]] = 0.
        assert not torch.isnan(i).any()

        # debias low-degree
        b_add = (gamma + 1) * self.W_add(i) + beta
        #b_add = self.W_add(i)

        # debias high-degree
        b_rev = (gamma + 1) * self.W_rev(i) + beta
        #b_rev = self.W_rev(i)

        mean_degree = torch.mean(degree.float())
        K = mean_degree * self.k
        R = torch.where(degree < K, torch.cuda.FloatTensor([1.]), torch.cuda.FloatTensor([0.]))


        #b_rev = b_add
        # compute constraints
        L_b = torch.sum(torch.norm((R*b_add)[idx], dim=1)) + torch.sum(torch.norm(((1-R)*b_rev)[idx], dim=1))
        L_b /= idx.shape[0]

        L_film = torch.sum(torch.norm(gamma[idx], dim=1)) + torch.sum(torch.norm(beta[idx], dim=1))
        L_film /= idx.shape[0]

        if not self.no_structural_contrast and not self.no_modulation:
            bias = self.omega * (R * b_add - (1-R) * b_rev)
        elif self.no_structural_contrast and not self.no_modulation:
            bias = self.omega * (b_add + b_rev)
        elif not self.no_structural_contrast and self.no_modulation:
            bias = 0
        elif self.no_structural_contrast and self.no_modulation:
            bias = 0

        output = output + bias

        if self.base == 3:
            output = self.lin(torch.cat([output, x], dim=1))
            output = h/torch.linalg.vector_norm(h, dim=1)

        features = {'L_b': L_b, 'L_film': L_film}
        return output, features


class Debias_v4(nn.Module):
    """
    modified DegFairGNN code
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            base,
            dataset,
            dim_d,
            omega,
            k,
            d_max,
            ver=2,
            dropout=0.5,
            nheads=1,
            structural_contrast=True,
            modulation=True,
            random_miss=False,
            miss=True,
            localization=True,
        ):
        super(Debias_v4, self).__init__()

        self.dim_M = dim_d
        self.out_channels = out_channels
        self.omega = omega
        self.d_max = (d_max+512) #0->dmax
        self.base = base
        self.dataset = dataset
        self.k = k
        self.dropout = dropout
        self.nheads = nheads
        self.structural_contrast = structural_contrast
        self.modulation = modulation
        self.random_miss = random_miss
        self.miss = miss
        self.localization = localization
        #self.w = args.w
        #self.sparse = args.sparse[]

        if ver == 1:
            self.r = Relation(in_channels, out_channels)
        else:
            self.r = Relationv2(in_channels, out_channels, not self.localization)
        self.g = Generator(in_channels)

        # self.weight = nn.Linear(in_channels, out_channels)
        if base == 1:
            self.nheads = 1
            self.conv = GCNConv(
                in_channels,
                out_channels,
                improved=False,
                cached=False,
                add_self_loops=True,
                normalize=True,
                bias=False,
            )
        elif base == 2:
            self.conv = SpGraphAttentionLayer(
                in_features=in_channels,
                out_features=out_channels,
                dropout=dropout,
                alpha=0.2,
            )
            # self.conv = GATConv(
            #     in_channels,
            #     out_channels,
            #     heads=self.nheads,
            #     concat=True,
            #     negative_slope=0.01,
            #     dropout=self.dropout,
            #     add_self_loops=True,
            #     edge_dim=None,
            #     bias=False,
            # )
            # self.a = nn.Parameter(torch.zeros(size=(1, 2*out_channels)))
            # nn.init.xavier_uniform_(self.a.data, gain=1.414)
            # self.special_spmm = SpecialSpmm()
            # self.dropout = nn.Dropout()
        elif base == 3:
            self.nheads = 1
            self.conv = SAGEConv(
                in_channels,
                out_channels,
                aggr = 'mean',
                normalize=False,
                root_weight=False,
                project=False,
                bias=False,
            )
            self.conv.lin_l.register_forward_hook(self.hook)
            self.lin = nn.Linear(out_channels + in_channels, out_channels)

        self.set_hook()

        self.W_gamma = nn.Parameter(torch.FloatTensor(self.dim_M, out_channels))
        self.W_beta = nn.Parameter(torch.FloatTensor(self.dim_M, out_channels))
        self.U_gamma = nn.Parameter(torch.FloatTensor(out_channels, out_channels))
        self.U_beta = nn.Parameter(torch.FloatTensor(out_channels, out_channels))
        self.b_gamma = nn.Parameter(torch.FloatTensor(1, out_channels))
        self.b_beta = nn.Parameter(torch.FloatTensor(1, out_channels))

        self.W_add = nn.Linear(out_channels, self.nheads * out_channels, bias=False)  # nheads was modified to 1 when base isn't 2
        self.W_rev = nn.Linear(out_channels, self.nheads * out_channels, bias=False)

        # Positional Encoding
        PE = np.array([
            [pos / np.power(10000, (i-i%2)/self.dim_M) for i in range(self.dim_M)]
            for pos in range(self.d_max)])

        PE[:, 0::2] = np.sin(PE[:, 0::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])
        self.PE = torch.as_tensor(PE, dtype=torch.float32)
        self.PE = self.PE.cuda()
        self.set_parameters()

    def set_hook(self):
        if self.base == 1:
            self.conv.lin.register_forward_hook(self.hook)
        elif self.base == 3:
            self.conv.lin_l.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.lin_feat = output

    def set_parameters(self):
        #nn.init.uniform_(self.m)
        nn.init.uniform_(self.W_gamma)
        nn.init.uniform_(self.W_beta)
        nn.init.uniform_(self.U_gamma)
        nn.init.uniform_(self.U_beta)
        nn.init.uniform_(self.b_gamma)
        nn.init.uniform_(self.b_beta)

        '''
            M_stdv = 1. / math.sqrt(self.M.size(1))
            self.M.data.uniform_(-M_stdv, M_stdv)

            b_stdv = 1. / math.sqrt(self.b.size(1))
            self.b.data.uniform_(-b_stdv, b_stdv)

            for m in self.modules():
                print(m.weight)
        '''

    def forward(self, x, adj, degree, idx, edge, head):
        neighbor = torch.mm(adj, x) / (degree + 1)
        relation_output = self.r(x, neighbor)
        if self.random_miss:
            h_s = self.g(relation_output)  # it just returns random tensor
        else:
            h_s = relation_output

        if self.base != 2:
            output = self.conv(x, adj)
            h = self.lin_feat  # self.lin_feat is the intermediate feature of conv layer. It is stored in hook function
        else:
            if head or not self.miss:
                output, h = self.conv(x, edge)
            else:
                output, h = self.conv(x, edge, mi=h_s)

        m_dv = torch.squeeze(self.PE[degree])
        if self.structural_contrast and not self.modulation:
            # if the model have structural contrast but no modulation, the model have different weights in high and low degree nodes
            # but degree encoding is not used
            m_dv = torch.zeros_like(m_dv)
        m_dv = m_dv.cuda()

        # version 1
        h = h * self.dim_M**0.5
        gamma = F.leaky_relu(torch.matmul((m_dv), self.W_gamma) + self.b_gamma) #
        beta = F.leaky_relu(torch.matmul((m_dv), self.W_beta) + self.b_beta) #

        # neighbor mean
        i = torch.spmm(adj, h.detach())
        i = i / degree
        i[torch.where(degree==0)[0]] = 0.
        assert not torch.isnan(i).any()

        # debias low-degree
        b_add = (gamma + 1) * self.W_add(i) + beta
        #b_add = self.W_add(i)

        # debias high-degree
        b_rev = (gamma + 1) * self.W_rev(i) + beta
        #b_rev = self.W_rev(i)

        mean_degree = torch.mean(degree.float())
        K = mean_degree * self.k
        R = torch.where(degree < K, torch.cuda.FloatTensor([1.]), torch.cuda.FloatTensor([0.]))


        #b_rev = b_add
        # compute constraints
        L_b = torch.sum(torch.norm((R*b_add)[idx], dim=1)) + torch.sum(torch.norm(((1-R)*b_rev)[idx], dim=1))
        L_b /= idx.shape[0]

        L_film = torch.sum(torch.norm(gamma[idx], dim=1)) + torch.sum(torch.norm(beta[idx], dim=1))
        L_film /= idx.shape[0]

        if self.structural_contrast and self.modulation:
            bias = self.omega * (R * b_add - (1-R) * b_rev)
        elif not self.structural_contrast and self.modulation:
            bias = self.omega * (b_add - b_rev)
        elif self.structural_contrast and not self.modulation:
            bias = self.omega * (R * b_add - (1-R) * b_rev)
        elif not self.structural_contrast and not self.modulation:
            bias = 0

        if not head and self.miss:
            if self.base == 1:
                h_s = self.conv.lin(h_s) / (degree + 1)
            elif self.base == 3:
                h_s = self.conv.lin_l(h_s) / (degree + 1)
            
            if self.base != 2:
                output = output + h_s
            # if self.base == 2, output is already added with h_s in SpGraphAttentionLayer

        if self.base == 3:
            output = self.lin(torch.cat([output, x], dim=1))
            output = h / (torch.linalg.vector_norm(h, dim=1).unsqueeze(-1) + EPS)

        output = output + bias

        features = {'L_b': L_b, 'L_film': L_film, 'relation_output': relation_output}
        # features = {'L_b':torch.tensor(0.).cuda(), 'L_film': torch.tensor(0.).cuda(), 'relation_output': relation_output}

        return output, features
