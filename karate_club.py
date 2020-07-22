import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import numpy as np
import networkx as nx

import dgl
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = F.relu(self.conv1(g, inputs))
        h = self.conv2(g, h)
        return h


def build_graph():
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
                    10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
                    25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
                    32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
                    33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
                    5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
                    24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
                    29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
                    31, 32])

    # Make edges bi-directional
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    graph = dgl.DGLGraph((u, v))
    return graph


def visualize_graph(graph: dgl.DGLGraph):
    import matplotlib.pyplot as plt
    G = graph.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()


def train():
    net = GCN(5, 5, 2)
    embed = nn.Embedding(34, 5)  # 34 nodes, each with feature dim 5

    G = build_graph()
    G.ndata['feat'] = embed.weight
    inputs = embed.weight
    labeled_nodes = torch.tensor([0, 33])
    labels = torch.tensor([0, 1])

    optimizer = torch.optim.Adam(itertools.chain(
        net.parameters(), embed.parameters()), lr=0.01)

    outputs = []
    for epoch in range(500):
        output = net(G, inputs)
        outputs.append(output.detach())
        loss = F.cross_entropy(output[labeled_nodes], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))


if __name__ == '__main__':
    train()
