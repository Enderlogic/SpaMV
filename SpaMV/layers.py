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
