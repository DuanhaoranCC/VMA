import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.glob import SumPooling, MaxPooling, AvgPooling
from dgl.nn.pytorch import GINConv
from torch_geometric.nn import BatchNorm, LayerNorm
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss = - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def mask(g, x, mask_rate=0.5):
    num_nodes = g.num_nodes()
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)
    mask_nodes = perm[: num_mask_nodes]

    return mask_nodes


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, num_layers):
        super(Encoder, self).__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.act = nn.ModuleList()
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                mlp = MLP(in_dim, hidden, out_dim)
            else:
                mlp = MLP(out_dim, hidden, out_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(BatchNorm(out_dim))
            self.act.append(nn.PReLU())
        self.pool = SumPooling()

    def forward(self, graph, h):
        output = []
        for i, layer in enumerate(self.ginlayers):
            h = layer(graph, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            output.append(self.pool(graph, h))

        return h, torch.cat(output, dim=1)


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def graph_show(graph):
    g = nx.Graph(dgl.to_networkx(graph))
    nx.draw(g, pos=nx.kamada_kawai_layout(g),
            node_color=graph.ndata["node_labels"].numpy(),
            node_size=300, width=4, with_labels=False)
    # nx.draw(g, pos=nx.spring_layout(g), node_color=graph.ndata["node_labels"].numpy(), node_size=50)
    plt.savefig(f'./M11.svg', dpi=800)
    plt.show()
    # plt.close()


class CMAE(nn.Module):
    def __init__(self, in_dim, out_dim, rate, hidden, t, layers):
        super(CMAE, self).__init__()
        self.encoder = Encoder(in_dim, out_dim, hidden, layers)
        # self.decoder = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, in_dim))
        self.contrast = Encoder(in_dim, out_dim, hidden, layers)
        self.proj_head = nn.Sequential(nn.Linear(out_dim * layers, 64 * layers), nn.ReLU(inplace=True),
                                       nn.Linear(64 * layers, 128))
        # self.pred_head = nn.Sequential(nn.Linear(out_dim * layers, 64 * layers), nn.ReLU(inplace=True),
        #                                nn.Linear(64 * layers, 128))

        self.rate = rate
        self.t = t
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.criterion = self.setup_loss_fn("sce")
        self.init_emb()

        # stop gradient
        for param in self.contrast.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.encoder.parameters()) + list(self.proj_head.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.contrast.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=1)
        else:
            raise NotImplementedError
        return criterion

    def cl_loss(self, x, x_aug, t=0.2):

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / t)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

    def forward(self, graph, feat):

        # MAE
        mask_nodes = mask(graph, feat, mask_rate=self.rate)
        # graph_show(graph.cpu(), 0)
        x = feat.clone()
        x[mask_nodes] = 0.0
        x[mask_nodes] += self.enc_mask_token

        h, gh = self.encoder(graph, x)

        # Contrastive
        _, ch = self.contrast(graph, feat)
        c_h = self.proj_head(ch)
        c_m = self.proj_head(gh)
        loss = self.cl_loss(c_h, c_m, self.t)
        # loss2 = self.cl_loss(c_m, c_h, self.t)
        # loss = cl_loss
        # embeds1 = c_h.clone()
        # embeds2 = c_m.clone()
        # embeds1 /= embeds1.norm(p=2, dim=1, keepdim=True)
        # embeds2 /= embeds2.norm(p=2, dim=1, keepdim=True)
        # align = align_loss(embeds1, embeds2, alpha=2)
        # unif = (uniform_loss(embeds1, t=2) + uniform_loss(embeds2, t=2)) / 2
        # print(align.item(), unif.item())
        return loss

    def get_embed(self, graph, feat):
        _, h = self.encoder(graph, feat)

        return h.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
