import torch
import torch.nn as nn
from src.utils.pad_tensor import pad_tensor
from torch.nn.parameter import Parameter
from torch import Tensor
import math

def compute_cross_attention(Xs, Ys, res_list):
    attention_x_list = []
    attention_y_list = []
    for x,y,s in zip(Xs, Ys, res_list):
        a_x = torch.softmax(s, dim=1)  # i->j
        a_y = torch.softmax(s, dim=0)  # j->i
        attention_x = torch.mm(a_x, y)
        attention_x_list.append(attention_x)
        attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
        attention_y_list.append(attention_y)
    return attention_x_list, attention_y_list

class cross_attention_layer(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, input_dim, output_dim):
        super(cross_attention_layer, self).__init__()
        self.d = output_dim
        self.A = torch.nn.Linear(input_dim, output_dim)

    def forward(self, Xs, Ys, weights):
        res_list=[]
        for X, Y, W in zip(Xs, Ys, weights):
            coefficients = torch.tanh(self.A(W))
            res = torch.matmul(X * coefficients, Y.transpose(0, 1))
            res = torch.nn.functional.softplus(res) - 0.5
            res_list.append(res)
        attention_x_list, attention_y_list = compute_cross_attention(Xs, Ys, res_list)
        return attention_x_list, attention_y_list, res_list

