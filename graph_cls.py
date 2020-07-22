import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl
import networkx as nx
import matplotlib.pyplot as plt
from dgl.data import MiniGCDataset
from dgl.nn.pytorch import GraphConv

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class GraphModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(GraphModel, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, g: dgl.DGLGraph):
        h1 = g.in_degrees().view(-1, 1).float()
        h2 = g.out_degrees().view(-1, 1).float()
        h = torch.cat([h1, h2], 1).to(device)
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h
        h = dgl.mean_nodes(g, 'h')
        return self.linear(h)


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        print(epoch, loss.item(), correct, total, 1.*correct/total)


def train(epoch):
    print('Epoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        print(epoch, train_loss/(1+batch_idx),
              correct, total, 1.*correct/total)


def test(epoch):
    print('Testing..')
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        print(epoch, test_loss/(1+batch_idx), correct, total, 1.*correct/total)


# Dataset
trainset = MiniGCDataset(1000, 10, 11)
testset = MiniGCDataset(100, 10, 11)
trainloader = DataLoader(trainset, batch_size=100,
                         shuffle=True, collate_fn=collate)
testloader = DataLoader(testset, batch_size=100,
                        shuffle=False, collate_fn=collate)

# Model
model = GraphModel(2, 128, trainset.num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
# optimizer = optim.SGD(model.parameters(), lr=0.001,
#                       momentum=0.9, weight_decay=5e-4)

for epoch in range(500):
    train(epoch)
    test(epoch)
