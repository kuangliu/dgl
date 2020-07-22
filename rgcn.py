import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.data import citation_graph

import networkx as nx


class RGCNLayer(nn.Module):
    def __init__(self, )
