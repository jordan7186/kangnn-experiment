from torch_geometric.nn import GCN, GAT, GraphSAGE, GIN
from src.efficient_kan import KAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import spmm


class KANGNN(torch.nn.Module):
    def __init__(
        self,
        in_feat,
        hidden_feat,
        out_feat,
        use_bias=False,
        input_embed=False,
        kan_layers=2,  # Number of KAN layers
        mp_layers=3,  # Number of message passing layers
    ):
        super().__init__()

        self.input_embed = input_embed
        self.module_list = nn.ModuleList()
        # First layers
        if self.input_embed:
            self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
            if kan_layers == 2:
                self.module_list.append(KAN([hidden_feat, hidden_feat, hidden_feat]))
            elif kan_layers == 1:
                self.module_list.append(KAN([hidden_feat, hidden_feat]))
            else:
                raise ValueError("Invalid number of KAN layers")
        else:
            if kan_layers == 2:
                self.module_list.append(KAN([in_feat, hidden_feat, hidden_feat]))
            elif kan_layers == 1:
                self.module_list.append(KAN([in_feat, hidden_feat]))
            else:
                raise ValueError("Invalid number of KAN layers")
        # Intermediate MP layers
        if mp_layers > 2:
            for _ in range(mp_layers - 2):
                if kan_layers == 2:
                    self.module_list.append(
                        KAN([hidden_feat, hidden_feat, hidden_feat])
                    )
                elif kan_layers == 1:
                    self.module_list.append(KAN([hidden_feat, hidden_feat]))
                else:
                    raise ValueError("Invalid number of KAN layers")
        # Last layer
        if mp_layers > 1:
            if kan_layers == 2:
                self.module_list.append(KAN([hidden_feat, hidden_feat, out_feat]))
            elif kan_layers == 1:
                self.module_list.append(KAN([hidden_feat, out_feat]))
            else:
                raise ValueError("Invalid number of KAN layers")

    def forward(self, x, adj):
        if self.input_embed:
            x = self.lin_in(x)
        for kan in self.module_list:
            x = kan(spmm(adj, x))
        return x


class KANonly(torch.nn.Module):
    def __init__(
        self,
        in_feat,
        hidden_feat,
        out_feat,
        use_bias=False,
        kan_layers=2,
        input_embed=False,
    ):
        super().__init__()
        self.input_embed = input_embed

        if self.input_embed:
            self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
            if kan_layers == 2:
                self.kan = KAN([hidden_feat, hidden_feat, out_feat])
            elif kan_layers == 1:
                self.kan = KAN([hidden_feat, out_feat])
        else:
            if kan_layers == 2:
                self.kan = KAN([in_feat, hidden_feat, out_feat])
            elif kan_layers == 1:
                self.kan = KAN([in_feat, out_feat])

    def forward(self, x):
        if self.input_embed:
            x = self.lin_in(x)
        x = self.kan(x)
        return x
