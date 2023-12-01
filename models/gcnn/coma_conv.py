# coding: UTF-8
"""
    @date:  2023.03.21  week12  星期2
    @func:  weights用lins替代.
    @ref:   https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cheb_conv.html#ChebConv
"""

import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.utils import remove_self_loops

def normal(tensor, mean, std):
    if tensor is not None:
        torch.nn.init.normal_(tensor, mean=mean, std=std)

class ChebConv_Coma(ChebConv):
    def __init__(self, in_channels, out_channels, K, normalization=None, bias=True):
        super(ChebConv_Coma, self).__init__(in_channels, out_channels, K, normalization, bias)

    def reset_parameters(self):
        # normal(self.lins, 0, 0.1)
        normal(self.bias, 0, 0.1)


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, norm, edge_weight=None):
        Tx_0 = x
        out = torch.matmul(Tx_0, self.lins[0])

        x = x.transpose(0,1)
        Tx_0 = x
        if self.lins.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            Tx_1_transpose = Tx_1.transpose(0, 1)
            out = out + torch.matmul(Tx_1_transpose, self.lins[1])

        for k in range(2, self.lins.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            Tx_2_transpose = Tx_2.transpose(0, 1)
            out = out + torch.matmul(Tx_2_transpose, self.lins[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1, 1) * x_j