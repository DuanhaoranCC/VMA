import torch
from sklearn.manifold import TSNE
from torch.optim import AdamW, Adam
from model import CG, CosineDecayScheduler
from torch_geometric import seed_everything
import numpy as np
import warnings
from scipy.linalg import inv
import yaml
import argparse
from eval import label_classification
from dataset import load_data
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')


def evaluate_cluster(embeds, y, n_label):
    Y_pred = KMeans(n_label, random_state=0).fit(embeds).predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    sil = silhouette_score(embeds, Y_pred)  # 越大:-1~1
    ch = calinski_harabasz_score(embeds, Y_pred)  # 越大
    db = davies_bouldin_score(embeds, Y_pred)  # 越小
    return nmi, ari, sil, ch, db


def plot_embeddings(embeddings, data):
    Y = data

    # emb_list = []
    # for k in range(Y.shape[0]):
    #     emb_list.append(embeddings[k])
    emb_list = embeddings

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
    plt.savefig("./Cora.svg", format='svg')
    plt.show()


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataname]

    for k, v in configs.items():
        if "lr" in k or "w" in k:
            v = float(v)
        setattr(args, k, v)
    return args


parser = argparse.ArgumentParser(description="VMA")
parser.add_argument("--dataname", type=str, default="Cora")
parser.add_argument("--cuda", type=int, default=0)
args = parser.parse_args()
args = load_best_configs(args, "config.yaml")
dataname = args.dataname
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

label_type = args.label
graph, feat, label, train_mask, val_mask, test_mask = load_data(dataname)
n_feat = feat.shape[1]
n_classes = np.unique(label).shape[0]
graph = graph.to(device)
label = label.to(device)
feat = feat.to(device)

# print(evaluate_cluster(normalize(feat.detach().cpu().numpy(), norm='l2'), label.cpu().numpy(),
#                        label.max().cpu().numpy() + 1))

adj = graph.adj().to_dense().cpu().numpy()
A_tilde = adj + np.identity(adj.shape[0])
D = A_tilde.sum(axis=1)
A_ = np.array(np.diag(D ** -0.5).dot(A_tilde).dot(np.diag(D ** -0.5)))
alpha = 1e-6
idx = np.arange(len(A_))

Lambda = np.identity(len(A_))
L = np.diag(D) - adj
P = inv(L + alpha * Lambda)


def train(space):
    seed_everything(0)
    p_label = torch.zeros(len(A_), len(A_))
    for i in range(len(A_)):
        m = np.argsort(P[:, i]).tolist()[::-1][:int(space['t1'])]
        p_label[i][m] = 1
    model = CG(n_feat, 256, space['p1'], space['p2'], space['rate'], 256, int(space['l']), space['t']).to(device)
    # scheduler
    # lr_scheduler = CosineDecayScheduler(args.lr, 100, args.epochs)
    # mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, args.epochs)
    # optimizer
    optimizer = Adam(model.trainable_parameters(), lr=space['lr'], weight_decay=space['w'])

    for epoch in range(1, int(space['epoch']) + 1):
        model.train()

        # update learning rate
        # lr = lr_scheduler.get(epoch - 1)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        # update momentum
        # mm = 1 - mm_scheduler.get(epoch - 1)
        mm = 0.9999
        loss = model(graph, feat, p_label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        # print(f"Epoch: {epoch}, Loss: {loss.item()}")
        # if epoch % 1000 == 0:
    model.eval()
    z1 = model.get_embed(graph, feat)
    # print(evaluate_cluster(normalize(z1.detach().cpu().numpy(), norm='l2'), label.cpu().numpy(),
    #                        label.max().cpu().numpy() + 1))
    # plot_embeddings(normalize(z1.detach().cpu().numpy(), norm='l2'), label.cpu().numpy())
    acc = label_classification(z1, train_mask, val_mask, test_mask,
                               label, label_type, name=dataname, device=device)
    space['acc'] = acc['Acc']['mean']
    print(space)
    return space['acc']

    # return {'loss': -round(space['acc'], 4), 'status': STATUS_OK}

    # print(f" Acc: {acc['Acc']['mean']}, Std: {round(acc['Acc']['std'], 4)}")


# trials = Trials()
# t = {
#     "lr": hp.choice('lr', [1e-6, 5e-6, 8e-6, 1e-5, 5e-5, 8e-5, 1e-4, 5e-4, 8e-4, 1e-3,
#                            5e-3, 8e-3]),
#     "w": hp.choice('w', [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
#     "p1": hp.quniform('p1', 0.0, 0.9, 0.1),
#     "p2": hp.quniform('p2', 0.1, 0.9, 0.1),
#     "rate": hp.quniform('rate', 0.1, 0.9, 0.1),
#     "t": hp.quniform('t', 0.1, 0.4, 0.1),
#     "t1": hp.quniform('t1', 1, 100, 1),
#     "l": hp.quniform('l', 1, 3, 1),
#     "epoch": hp.quniform('epoch', 100, 10000, 100)
# }
# best = fmin(train, space=t, algo=tpe.suggest, max_evals=200, trials=trials)
# print(best)

# Cora 54.58, 44.70
train({'epoch': 4000, 'lr': 0.001, 'p1': 0.5, 'p2': 0.8, 'rate': 0.9,
       't': 0.2, 't1': 2, 'l': 2, 'w': 0.0005, 'acc': 0.838})
# Com(0.5474722748769139, 0.3408875428047783, 0.21571127
# train({'epoch': 4000, 'l': 1, 'lr': 0.001, 'p1': 0.7, 'p2': 0.4,
#        'rate': 0.7, 't': 0.2, 't1': 23, 'w': 5e-5, 'acc': 0.9045})
# Phy(0.525269291591095, 0.3054629341399983, 0.10640577, 2114.905296869902, 2.8239815413696965)
# train({'epoch': 4000, 'l': 1, 'lr': 0.008, 'p1': 0.6, 'p2': 0.7,
#        'rate': 0.3, 't': 0.2, 't1': 28, 'w': 0.001, 'acc': 0.9551})
# PubMed 24.795, 22.287
# train({'epoch': 4000, 'l': 1, 'lr': 0.0005, 'p1': 0.8, 'p2': 0.5,
#        'rate': 0.6, 't': 0.3, 't1': 60, 'w': 5e-5, 'acc': 0.828})
# Photo 68.64, 60.45, 31.51
# train({'epoch': 4000, 'l': 1, 'lr': 0.0005, 'p1': 0.0, 'p2': 0.9,
#        'rate': 0.9, 't': 0.2, 't1': 27, 'w': 0.0001, 'acc': 0.9352})
# WikiCS 48.817, 42.63, 16.108
# train({'epoch': 6000, 'l': 2, 'lr': 0.001, 'p1': 0.6, 'p2': 0.6,
#        'rate': 0.7, 't': 0.2, 't1': 79, 'w': 0.0, 'acc': 0.8017})
# CS 76.085, 60.879, 25.4
# train({'epoch': 6000, 'l': 1, 'lr': 0.008, 'p1': 0.2, 'p2': 0.9,
#        'rate': 0.5, 't': 0.2, 't1': 13, 'w': 0.0001, 'acc': 0.9279})
# DBLP
# train({'epoch': 1000, 'l': 2, 'lr': 5e-5, 'p1': 0.9, 'p2': 0.5,
#        'rate': 0.4, 't': 0.4, 't1': 24, 'w': 1e-5, 'acc': 0.7902})

# train({'epoch': 5000, 'lr': 7e-5, 'p1': 0.6, 'p2': 0.9, 'rate': 0.8, 't1': 2, 'l': 2,
#        't': 0.2, 'w': 0.0, 'acc': 0.708})

# train({'epoch': 4400, 'l': 1, 'lr': 0.008, 'p1': 0.7, 'p2': 0.9,
#        'rate': 0.2, 't': 0.1, 't1': 3.0, 'w': 0.001, 'acc': 0.728})

# for i in [1, 5, 10, 20, 30]:
#     train({'epoch': 6000, 'l': 2, 'lr': 0.001, 'p1': 0.6, 'p2': 0.6,
#            'rate': 0.7, 't': 0.2, 't1': i, 'w': 0.0, 'acc': 0.8017})
