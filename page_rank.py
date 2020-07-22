import dgl
import torch
import networkx as nx
import dgl.function as fn
import matplotlib.pyplot as plt


N = 100  # number of nodes
DAMP = .5
K = 10


def build_graph():
    g = nx.erdos_renyi_graph(N, 0.1)
    g = dgl.DGLGraph(g)
    #  nx.draw(g.to_networkx())
    #  plt.show()
    return g


def assign_feature(g: dgl.DGLGraph):
    g.ndata['pv'] = torch.ones(N) / N
    g.ndata['deg'] = g.out_degrees(g.nodes()).float()


def page_rank_message(edges):
    return {'pv': edges.src['pv'] / edges.src['deg']}


def page_rank_reduce(nodes):
    msgs = torch.sum(nodes.mailbox['pv'], dim=1)
    pv = (1 - DAMP) / N + DAMP * msgs
    return {'pv': pv}


def page_rank_naive(g: dgl.DGLGraph):
    # Send out message along all edges
    for u, v in zip(*g.edges()):
        g.send((u, v))
    # Receive message and update pagerank value
    for v in g.nodes:
        g.recv(v)


def page_rank_batch(g: dgl.DGLGraph):
    g.send(g.edges())
    g.recv(g.nodes())


def page_rank_level2(g: dgl.DGLGraph):
    g.update_all()


def page_rank_builtin(g: dgl.DGLGraph):
    g.ndata['pv'] = g.ndata['pv'] / g.ndata['deg']
    g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                 reduce_func=fn.sum(msg='m', out='sum'))
    g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['sum']


if __name__ == '__main__':
    g = build_graph()
    assign_feature(g)
    g.register_message_func(page_rank_message)
    g.register_reduce_func(page_rank_reduce)
    for k in range(K):
        page_rank_builtin(g)
    print(g.ndata['pv'])
