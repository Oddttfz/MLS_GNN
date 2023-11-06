import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        self.n_layers =n_layers
        for i in range(n_layers -1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features,middle=False):
        h = features
        y = features
        middle_feats=[]
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g.to("cuda:0"), h)
            y = self.layers[i](self.g.to("cuda:0"), y)############
            middle_feats.append(y)
            y = F.relu(y)
        if middle:
            return h,middle_feats
        return h
    '''
    def forward(self, inputs, middle=False):
        h = inputs
        middle_feats = []
        for l in range(self.n_layers):
            h = self.layers[l](self.g.to(torch.device("cuda:0")), h)
            middle_feats.append(h)
            h = F.relu(h)
        logits = self.layers[-1](self.g.to(torch.device("cuda:0")), h)
        middle_feats.append(logits)
        if middle:
            return logits, middle_feats
        return logits
    '''

class My_GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(My_GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h


class ogb_GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear=False):
        super().__init__()
        self.g = g
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(GraphConv(in_hidden, out_hidden, "both", bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_hidden))

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, feat):
        h = feat
        h = self.dropout0(h)
        for i in range(self.n_layers):
            conv = self.convs[i](self.g, h)
            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv
            if i < self.n_layers - 1:
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h


class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lr1 = nn.Linear(hidden_dim, output_dim)
        # self.lr3 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.lr1(self.dropout(x)))
        # return self.lr3(self.dropout(x))