import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
from torch.optim import Adam
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl
import networkx as nx
from model import CMAE, CosineDecayScheduler, LogReg
from torch_geometric import seed_everything
import numpy as np
import warnings
import yaml
import argparse
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from dataset import load_graph_classification_dataset, train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

warnings.filterwarnings('ignore')


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataname]

    for k, v in configs.items():
        if "lr" in k or "w" in k:
            v = float(v)
        setattr(args, k, v)
    return args


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []

    kf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std


def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


seed_everything(0)
parser = argparse.ArgumentParser(description="CMAE")
parser.add_argument("--dataname", type=str, default="MSRC_21")
parser.add_argument("--cuda", type=int, default=0)
args = parser.parse_args()
args = load_best_configs(args, "config.yaml")
dataname = args.dataname
# s = time.perf_counter()
graphs, (n_feat, num_classes), Y = load_graph_classification_dataset(dataname)
train_idx = torch.arange(len(graphs))
batch_size = 256
train_sampler = SubsetRandomSampler(train_idx)
train_loader = GraphDataLoader(graphs, collate_fn=collate_fn,
                               batch_size=batch_size, shuffle=True)
eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=batch_size,
                              shuffle=True)

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')


def train(space):
    seed_everything(0)
    model = CMAE(n_feat, 32, space['rate'], args.out_hidden, space['t'], int(space['l'])).to(device)
    optimizer = Adam(model.trainable_parameters(), lr=space['lr'], weight_decay=space['w'])
    # lr_scheduler = CosineDecayScheduler(args.lr, args.warmup, args.epochs)
    # mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, args.epochs)
    # total_params = sum(p.numel() for p in model.parameters())
    # print("Number of parameter: %.2fM" % (total_params/1e6))
    for epoch in range(1, int(space['epoch']) + 1):
        model.train()
        # update momentum
        # mm = 1 - mm_scheduler.get(epoch - 1)
        mm = 0.9999
        # update learning rate
        # lr = lr_scheduler.get(epoch - 1)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        for g, _ in train_loader:
            g = g.to(device)
            loss = model(g, g.ndata["attr"])
            # print(f"Epoch: {epoch}, Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_network(mm)
    x_list = []
    y_list = []
    model.eval()
    # i = 0
    for g, label in eval_loader:
        g = g.to(device)
        z1 = model.get_embed(g, g.ndata['attr'])
        y_list.append(label.numpy())
        x_list.append(z1.detach().cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    # print(args)
    space['acc'] = test_f1
    print(space)
    # return test_f1
    # print(f"Acc: {test_f1}, Std: {test_std}")
    return {'loss': -round(space['acc'], 4), 'status': STATUS_OK}


# trials = Trials()
# t = {
#     "lr": hp.choice('lr', [1e-6, 5e-6, 8e-6, 1e-5, 5e-5, 8e-5, 1e-4, 5e-4, 8e-4, 1e-3,
#                            5e-3, 8e-3]),
#     "w": hp.choice('w', [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
#     "rate": hp.quniform('rate', 0.1, 0.9, 0.1),
#     "t": hp.quniform('t', 0.1, 0.4, 0.1),
#     "l": hp.quniform('l', 1, 3, 1),
#     "epoch": hp.quniform('epoch', 100, 1000, 100)
# }
# best = fmin(train, space=t, algo=tpe.suggest, max_evals=200, trials=trials)
# print(best)

# PROTEINS 76.31+-0.55
# train({'epoch': 500, 'l': 3, 'lr': 0.005, 'rate': 0.6,
#        't': 0.3, 'w': 5e-5, 'acc': 0.77087})
# COLLAB 80.76+-0.3
# train({'epoch': 700, 'l': 2, 'lr': 0.008, 'rate': 0.6,
#        't': 0.1, 'w': 0.0, 'acc': 0.8078})
# MUTAG 91.06+-2.5
# train({'epoch': 500, 'l': 2, 'lr': 8e-5, 'rate': 0.5,
#        't': 0.3, 'w': 0.001, 'acc': 0.9196})
# IMDB-BINARY75.59+-0.498
# train({'epoch': 100, 'l': 3, 'lr': 0.008, 'rate': 0.4,
#        't': 0.4, 'w': 0.001, 'acc': 0.7649})
# IMDB-MULTI 52.51+-0.179
# train({'epoch': 900, 'l': 2, 'lr': 0.0008, 'rate': 0.7,
#        't': 0.4, 'w': 5e-5, 'acc': 0.5273})
# NCI1 81.02+-0.0
# train({'epoch': 200, 'l': 3, 'lr': 0.0001, 'rate': 0.2,
#        't': 0.2, 'w': 0.001, 'acc': 0.81.02})
# DD 79.38+-0.0
# train({'epoch': 200, 'l': 2, 'lr': 0.0005, 'rate': 0.8,
#        't': 0.3, 'w': 0.0005, 'acc': 0.79538})
# REDDIT-BINARY
# train({'epoch': 500, 'l': 1, 'lr': 0.008, 'rate': 0.8,
#        't': 0.2, 'w': 0.0005, 'acc': 0.913})
# 0.8 0.1
# MSRC
# train({'epoch': 100, 'l': 3, 'lr': 0.005, 'rate': 0.5,
#        't': 0.3, 'w': 5e-5, 'acc': 0.90416})
# ENZYMES
# train({'epoch': 800, 'l': 3, 'lr': 0.001, 'rate': 0.1,
#        't': 0.2, 'w': 0.0, 'acc': 0.61166})


a = []
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    train({'epoch': 100, 'l': 3, 'lr': 0.005, 'rate': i,
           't': 0.3, 'w': 5e-5, 'acc': 0.90416})
# print(np.array(a).mean(), np.array(a).std())
# a = []
# for i in range(10):
#     a.append(train({'epoch': 500, 'l': 2, 'lr': 8e-5, 'rate': i,
#        't': 0.3, 'w': 0.001, 'acc': 0.9196}))
# print(np.array(a).mean(), np.array(a).std())

