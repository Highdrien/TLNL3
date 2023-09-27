import torch
import torch.nn.functional as F


class Model():
    def __init__(self, embedding_dim, context_length, hidden_layer, vocab_size):
        # first layer
        self.W = torch.randn((embedding_dim * context_length, hidden_layer), requires_grad=True)
        self.b1 = torch.randn((1, hidden_layer), requires_grad=True)
        # second layer
        self.U = torch.randn((hidden_layer, vocab_size), requires_grad=True)
        self.b2 = torch.randn((1, vocab_size), requires_grad=True)

    def forward(self, x):
        h = F.relu(x @ self.W + self.b1)
        y = h @ self.U + self.b2
        return F.softmax(y, dim=1)

    def parameters(self):
        return [self.W, self.b1, self.U, self.b2]