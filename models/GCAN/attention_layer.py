import torch
import torch.nn as nn


class cross_attention_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cross_attention_layer, self).__init__()
        self.d = output_dim
        self.A = torch.nn.Linear(input_dim, output_dim)
        self.B = torch.nn.Linear(input_dim, output_dim)

    def _forward(self, X, Y, weights_avg,weights_max):
        assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
        coefficients = torch.tanh(self.A(weights_avg))
        coefficients_B = torch.tanh(self.B(weights_avg))
        # coefficients = torch.nn.functional.leaky_relu(self.A(weights_avg))
        coefficients_B = torch.nn.functional.leaky_relu(self.B(weights_avg))
        Y = Y*coefficients_B
        res = torch.matmul(X * coefficients, Y.transpose(0, 1))
        res = torch.nn.functional.softplus(res) - 0.5
        # res = torch.nn.functional.leaky_relu(res) - 0.5
        return res

    def forward(self, Xs, Ys, Ws_avg, Ws_max):
        return [self._forward(X, Y, W_avg, W_max) for X, Y, W_avg, W_max in zip(Xs, Ys, Ws_avg, Ws_max)]
