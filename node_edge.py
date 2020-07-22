import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt


def build_graph():
    G1 = nx.petersen_graph()
    G2 = dgl.DGLGraph(G1)

    plt.subplot(121)
    nx.draw(G1, with_labels=True)
    plt.subplot(122)
    nx.draw(G2.to_networkx(), with_labels=True)
    plt.show()


def constructor():
    u = torch.tensor([0, 0, 0, 0, 0])
    v = torch.tensor([1, 2, 3, 4, 5])
    G = dgl.DGLGraph((u, v))
    nx.draw(G.to_networkx(), with_labels=True)
    plt.show()

    G2 = dgl.DGLGraph((0, v))
    nx.draw(G2.to_networkx(), with_labels=True)
    plt.show()

    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    G3 = dgl.DGLGraph(edges)
    nx.draw(G3.to_networkx(), with_labels=True)
    plt.show()


def assign_feature():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    g.add_edges(list(range(1, 10)), 0)

    x = torch.randn(10, 3)
    g.ndata['x'] = x
    g.edata['w'] = torch.randn(9, 2)
    print(g.edge_id(1, 0))


if __name__ == '__main__':
    # build_graph()
    # constructor()
    assign_feature()
