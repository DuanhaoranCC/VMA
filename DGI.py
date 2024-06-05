import os.path as osp

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from dataset import load_data
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import torch_geometric.transforms as T
from torch_geometric.datasets import WikiCS, CitationFull, Amazon, Coauthor


def evaluate_cluster(embeds, y, n_label):
    Y_pred = KMeans(n_label, random_state=0).fit(embeds).predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    sil = silhouette_score(embeds, Y_pred)  # 越大:-1~1
    ch = calinski_harabasz_score(embeds, Y_pred)  # 越大
    db = davies_bouldin_score(embeds, Y_pred)  # 越小
    return nmi, ari, sil, ch, db


def plot_embeddings(embeddings, data):
    Y = data.y.detach().cpu().numpy()

    emb_list = []
    for k in range(Y.shape[0]):
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(Y.shape[0]):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)
    plt.figure()
    # ax = Axes3D(fig)
    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s=10, alpha=0.7)
    # plt.legend()
    plt.savefig("DGI_Freebase_mwm.svg", format='svg', dpi=600)
    # plt.close()
    # plt.clf()
    plt.show()


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


# dataset = Amazon(root='./', name='computers', transform=T.NormalizeFeatures())
dataset = WikiCS(root='./Wiki')
data = dataset[0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DeepGraphInfomax(
    hidden_channels=256, encoder=Encoder(data.x.size(1), 256),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
data = data.to(device)

def train():
    model.train()
    # total_params = sum(p.numel() for p in model.parameters())
    # print("Number of parameter: %.2fM" % (total_params / 1e6))
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(data.x, data.edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(1, 501):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
z, _, _ = model(data.x, data.edge_index)
print(evaluate_cluster(normalize(z.detach().cpu().numpy(), norm='l2'), data.y.cpu().numpy(),
                           data.y.max().cpu().numpy() + 1))