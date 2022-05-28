import argparse
import numpy as np
import yaml
from yaml import SafeLoader
from scipy.linalg import inv
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric import seed_everything
from model import Encoder, Model
from eval import label_classification
import warnings

warnings.filterwarnings('ignore')

seed = 35456
seed_everything(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='DBLP')
parser.add_argument('--config', type=str, default='config.yaml')
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()

cfg = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

num_hidden = cfg['num_hidden']
num_proj_hidden = cfg['num_proj_hidden']
activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'elu': nn.ELU()})[cfg['activation']]
base_model = ({'GCNConv': GCNConv})[cfg['base_model']]
num_layers = cfg['num_layers']
t = cfg['t']
weight_decay = cfg['weight_decay']
lr = cfg['learning_rate']
p1 = cfg['p1']
p2 = cfg['p2']
tau = cfg['tau']
Epoch = cfg['epoch']
label_type = 1
alpha = 1e-6
if args.dataset == "Photo" or args.dataset == "Computers":
    dataset = Amazon(root='./', name=args.dataset, transform=T.NormalizeFeatures())
if args.dataset == "DBLP":
    dataset = CitationFull(root='./', name=args.dataset)
if args.dataset == "Cora" or args.dataset == "PubMed":
    dataset = Planetoid(root='./', name=args.dataset)
    label_type = 0
    if args.dataset == "Cora":
        alpha = 1e-6
    else:
        alpha = 1e-5
if args.dataset == "CiteSeer":
    dataset = Planetoid(root='./', name=args.dataset, transform=T.NormalizeFeatures())
    label_type = 0
if args.dataset == "CS":
    dataset = Coauthor(root='./', name=args.dataset, transform=T.NormalizeFeatures())

data = dataset[0]
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

adj = to_dense_adj(data.edge_index).cpu().detach().numpy()[0]

A_tilde = adj + np.identity(adj.shape[0])
D = A_tilde.sum(axis=1)
A_ = np.array(np.diag(D ** -0.5).dot(A_tilde).dot(np.diag(D ** -0.5)))

idx = np.arange(len(A_))

Lambda = np.identity(len(A_))
L = np.diag(D) - adj
P = inv(L + alpha * Lambda)
p_label = torch.zeros(len(A_), len(A_)).to(device)
for i in range(len(A_)):
    m = np.argsort(P[:, i]).tolist()[::-1][:t]
    p_label[i][m] = 1


def train(model: Model, x, edge_index, optimizer, label, tau):
    model.train()
    optimizer.zero_grad()

    z1 = model(x, edge_index)
    z2 = model(x, edge_index)
    loss = model.loss(z1, z2, label, tau=tau)

    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers, dp1=p1, dp2=p2).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, Epoch + 1):
        loss = train(model, data.x, data.edge_index, optimizer, label=p_label, tau=tau)
        if epoch % 10 == 0:
            model.eval()
            z = model(data.x, data.edge_index)
            acc = label_classification(z, data, label_type)['Acc']
            print(f"Epoch:{epoch}, {loss}, {acc}")
