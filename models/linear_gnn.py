import torch.nn as nn


class LinearGNN(nn.Module):
    def __init__(self, model, nfeat, dim1, dim2, n_layer, nclass, dropout, **kwargs):
        super(LinearGNN, self).__init__()

        self.fc_in = nn.Linear(nfeat, dim1)
        self.gnn = model(dim1, dim2, n_layer, nclass, dropout, **kwargs)
    
    def forward(self, x, adj):
        x = self.fc_in(x)
        x = self.gnn(x, adj)
        return x
