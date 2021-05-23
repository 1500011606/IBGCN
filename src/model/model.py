import numpy as np 
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gcn_conv import BatchGCNConv

class Basic_Model(nn.Module):
    """Some Information about Basic_Model"""
    def __init__(self, args):
        super(Basic_Model, self).__init__()
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

        self.args = args

    def forward(self, data, adj):
        N = adj.shape[0]

        x = data.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]

        x = x + data
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

 
    def feature(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        return x

class FH(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.fH["in_channel"], args.fH["hidden_channel"])
        self.fc2 = nn.Linear(args.fH["hidden_channel"], args.fH["hidden_channel"])
        self.fc3 = nn.Linear(args.fH["hidden_channel"], args.fH["out_channel"])

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

class FL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.fL["in_channel"], args.fL["hidden_channel"])
        self.fc2 = nn.Linear(args.fL["hidden_channel"], args.fL["hidden_channel"])
        self.fc3 = nn.Linear(args.fL["hidden_channel"], args.fL["out_channel"])

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

class Integrated_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.FH = FH(args)
        self.FL = FL(args)
        self.GH = Basic_Model(args)
        self.GL = Basic_Model(args)
        self.fc = nn.Linear(args.y_len * 2, args.y_len)

    def forward(self, data, adj):
        N = adj.shape[0]
        # print("epang model input: ", data.x.shape)
        high_flow = self.FH(data.x)                                   # [bs * N, args.fH["out_channel"]]
        eigen_flow = self.FL(data.x)                                  # [bs * N, args.fL["out_channel"]]
        # print("epang eigen_flow: ", eigen_flow.shape)
        high_pred = self.GH(high_flow, adj)                           # [bs * N, args.y_len]
        eigen_pred = self.GL(eigen_flow, adj)                         # [bs * N, args.y_len]
        # print("epang high_pred: ", type(high_pred), high_pred.shape)
        total_pred = F.elu(self.fc(torch.cat([high_pred, eigen_pred],dim=1))) # [bs, N, args.y_len]
        # print("epang total_pred: ", total_pred.shape)
        return total_pred, high_flow, eigen_flow

