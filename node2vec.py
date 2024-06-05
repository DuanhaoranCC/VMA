import os.path as osp
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from torch_geometric.datasets import Amazon, WikiCS
from torch_geometric.nn import Node2Vec


def evaluate_cluster(embeds, y, n_label):
    Y_pred = KMeans(n_label, random_state=0).fit(embeds).predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    sil = silhouette_score(embeds, Y_pred)  # 越大:-1~1
    ch = calinski_harabasz_score(embeds, Y_pred)  # 越大
    db = davies_bouldin_score(embeds, Y_pred)  # 越小
    return nmi, ari, sil, ch, db


def main():
    dataset = Amazon(root='./', name='computers')
    # dataset = WikiCS(root='./Wiki')
    data = dataset[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
    ).to(device)

    num_workers = 0 if sys.platform.startswith('win') else 4
    loader = model.loader(batch_size=128, shuffle=True,
                          num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(1, 101):
        loss = train()
        # acc = test()
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
    z = model(torch.arange(data.num_nodes, device=device))
    print(evaluate_cluster(normalize(z.detach().cpu().numpy(), norm='l2'), data.y.cpu().numpy(),
                           data.y.max().cpu().numpy() + 1))


if __name__ == "__main__":
    main()
