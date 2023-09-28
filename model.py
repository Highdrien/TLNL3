from typing import List

import torch
import torch.nn.functional as F


class Model():
    def __init__(self, embedding_dim: int, context_length: int, hidden_layer: int, vocab_size: int):
        # first layer
        self.W = torch.randn((embedding_dim * context_length, hidden_layer), requires_grad=True)
        self.b1 = torch.randn((1, hidden_layer), requires_grad=True)
        # second layer
        self.U = torch.randn((hidden_layer, vocab_size), requires_grad=True)
        self.b2 = torch.randn((1, vocab_size), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x @ self.W + self.b1)
        y = h @ self.U + self.b2
        # return F.softmax(y, dim=1)    # ne pas mettre softmax pour la Cross sEntropy
        return y

    def parameters(self) -> List[torch.Tensor]:
        return [self.W, self.b1, self.U, self.b2]
    
    def to(self, device: torch.device) -> None:
        self.W.to(device)
        self.b1.to(device)
        self.U.to(device)
        self.b2.to(device)