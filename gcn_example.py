import os.path as osp
import argparse

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import HeteroDataset
import jittor_geometric.transforms as T
from jittor_geometric.nn import GCNConv
# add by lusz
import time

jt.flags.use_cuda = 1


dataset = 'roman_empire'
dataset = HeteroDataset("./data", dataset, transform=T.NormalizeFeatures())
data = dataset[0]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 64, cached=True)
        self.conv2 = GCNConv(64, dataset.num_classes, cached=True)

    def execute(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = nn.relu(self.conv1(x, edge_index, edge_weight))
        x = nn.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return nn.log_softmax(x, dim=1)


model, data = Net(), data
optimizer = nn.Adam([
    dict(params=model.parameters(), weight_decay=5e-4),
], lr=0.01)


def train(run):
    model.train()
    pred = model()[data.train_mask[run]]
    label = data.y[data.train_mask[run]]
    loss = nn.nll_loss(pred, label)
    jt.sync_all(True)
    optimizer.step(loss)


def test(run):
    model.eval()
    logits = model()
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        mask = mask[run]
        y_ = data.y[mask]
        logits_ = logits[mask]
        pred, _ = jt.argmax(logits_, dim=1)
        acc = pred.equal(y_).sum().item() / mask.sum().item()
        accs.append(acc)

    return accs

best_val_acc = test_acc = 0
start = time.time()

for run in range(10):
    for epoch in range(1, 201):
        train(run)
        train_acc, val_acc, tmp_test_acc = test(run)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

jt.sync_all(True)
end = time.time()
print("epoch_time"+str(end-start))

