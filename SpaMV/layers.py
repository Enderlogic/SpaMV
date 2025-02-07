import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=1, interpretable=True):
        super().__init__()
        self.out_dim = out_dim
        self.interpretable = interpretable
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads)
        self.batch_norm1 = BatchNorm(hidden_dim * heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, out_dim * 2, heads=heads, concat=False)
        self.batch_norm2 = BatchNorm(out_dim, affine=False)

    def forward(self, x, edge_index):
        # x = F.relu(self.conv1(x, edge_index))
        # output = self.conv2(x, edge_index)
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        output = self.conv2(x, edge_index)
        if self.interpretable:
            # return output[:, :self.out_dim], output[:, self.out_dim:]
            return self.batch_norm2(output[:, :self.out_dim]), F.sigmoid(output[:, self.out_dim:])
        else:
            # return output[:, :self.out_dim], output[:, self.out_dim:]
            return self.batch_norm2(output[:, :self.out_dim]), output[:, self.out_dim:]


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_size, out_dim, recon_type='nb'):
        super().__init__()
        # setup the two linear transformations used
        self.recon_type = recon_type
        self.fc1 = nn.Linear(z_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_dim)
        if recon_type == 'zinb':
            self.zi_logits = nn.Linear(hidden_size, out_dim)

    def forward(self, z):
        hidden = F.relu(self.fc1(z))
        if self.recon_type == 'zinb':
            return F.softplus(self.fc2(hidden)), self.zi_logits(hidden)
        else:
            return F.softplus(self.fc2(hidden))


class Decoder_zi_logits(nn.Module):
    def __init__(self, z_dim, hidden_size, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_dim)

    def forward(self, z):
        hidden = F.relu(self.fc1(z))
        return self.fc2(hidden)


class Distinguished_Decoder(nn.Module):
    def __init__(self, zp_dims, hidden_dim, data_dims, recon_types, omics_names, interpretable):
        super(Distinguished_Decoder, self).__init__()
        self.recon_types = recon_types
        self.omics_names = omics_names
        self.interpretable = interpretable
        self.fc1 = nn.ModuleDict()
        self.fc2 = nn.ModuleDict()
        self.w = nn.ParameterDict()
        for i in range(len(data_dims)):
            for j in range(len(data_dims)):
                if i != j:
                    name = "from_" + omics_names[i] + "_to_" + omics_names[j]
                    if interpretable:
                        self.w[name] = nn.Parameter(torch.randn(zp_dims[i], data_dims[j]))
                    else:
                        self.fc1[name] = nn.Linear(zp_dims[i], hidden_dim)
                        self.fc2[name] = nn.Linear(hidden_dim, data_dims[j])
    def forward(self, zps):
        output = {}
        for i in range(len(self.omics_names)):
            for j in range(len(self.omics_names)):
                if i != j:
                    name = "from_" + self.omics_names[i] + "_to_" + self.omics_names[j]
                    if self.interpretable:
                        output[name] = zps[i] @ F.softplus(self.w[name])
                    else:
                        hidden = F.relu(self.fc1[name](zps[i]))
                        output[name] = F.softplus(self.fc2[name](hidden))
                    output[name] = output[name].clamp(min=1e-10, max=1e8)
        return output
