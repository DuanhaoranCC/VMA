import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LayerNorm, BatchNorm


# Code reference resources https://github.com/flyingtango/DiGCL,
#                          https://github.com/CRIPAC-DIG/GRACE
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


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2, dp1=0.1, dp2=0.1):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.dp1 = dp1
        self.dp2 = dp2
        self.conv = [base_model(in_channels, 2*out_channels)]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2*out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.ly1 = LayerNorm(2 * out_channels)
        self.ly2 = LayerNorm(out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        x = F.dropout(x, self.dp1, training=self.training)
        
        x = self.conv[0](x, edge_index)      
        x = F.dropout(x, self.dp2, training=self.training)
        x = self.ly1(x)
        x = self.activation(x)

        x = self.conv[1](x, edge_index)
        x = self.ly2(x)
        return self.activation(x)


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.bn = nn.BatchNorm1d(num_proj_hidden)


    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.bn(self.fc1(z)))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, label, tau):
        def f(x): return torch.exp(x / tau)
        z11 = self.sim(z1, z1)
        z12 = self.sim(z1, z2)
        refl_sim = f(z11)
        between_sim = f(z12)
        return -torch.log(
            (torch.mul(between_sim, label).sum(1) + torch.mul(refl_sim, label).sum(1) - refl_sim.diag())
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, label, tau, mean: bool = True):

        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2, label, tau)
        l2 = self.semi_loss(h2, h1, label, tau)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
