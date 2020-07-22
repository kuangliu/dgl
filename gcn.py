import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.data import citation_graph

import networkx as nx


class GCNLayer(nn.Module):
    '''GCNLayer performs message passing on all nodes then apply a linear layer.'''

    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, x):
        with g.local_scope():
            g.ndata['h'] = x
            g.update_all(fn.copy_src(src='h', out='m'),
                         fn.sum(msg='m', out='h'))
            h = g.ndata['h']
            return self.linear(h)


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(1433, 16)
        self.layer2 = GCNLayer(16, 7)

    def forward(self, g, x):
        out = F.relu(self.layer1(g, x))
        out = self.layer2(g, out)
        return out


def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    outputs = net(g, inputs)
    loss = F.cross_entropy(outputs[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.item()
    _, predicted = outputs[train_mask].max(1)
    total = labels[train_mask].size(0)
    correct = (predicted == labels[train_mask]).sum().item()
    print(loss, correct, total, 1.*correct/total)


def test(epoch):
    print('Testing..')
    net.eval()
    with torch.no_grad():
        outputs = net(g, inputs)
        _, predicted = outputs[test_mask].max(1)
        total = labels[test_mask].size(0)
        correct = (predicted == labels[test_mask]).sum().item()
        print(correct, total, 1.*correct/total)


# Data
data = citation_graph.load_cora()
inputs = torch.FloatTensor(data.features)
labels = torch.LongTensor(data.labels)
test_mask = torch.BoolTensor(data.train_mask)
train_mask = torch.BoolTensor(data.test_mask)
g = dgl.DGLGraph(data.graph)

# Model
net = GCN()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

for epoch in range(200):
    train(epoch)
    test(epoch)
