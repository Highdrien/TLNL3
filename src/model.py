from typing import List
import os

import torch
import torch.nn.functional as F


class Model():
    def __init__(self, embedding_dim: int, context_length: int, hidden_layer: int, vocab_size: int, learn_embedding: bool):
        # Embedding layer
        if learn_embedding:
            self.E = torch.randn((vocab_size, embedding_dim))
        # first Feed Forward NN
        self.W = torch.randn((embedding_dim * context_length, hidden_layer), requires_grad=True)
        self.b1 = torch.randn((1, hidden_layer), requires_grad=True)
        # second Feed Forward NN
        self.U = torch.randn((hidden_layer, vocab_size), requires_grad=True)
        self.b2 = torch.randn((1, vocab_size), requires_grad=True)

        self.context_length = context_length
        self.learn_embedding = learn_embedding
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ pass x throught the model and return the output (without softmax) """
        if self.learn_embedding:
            x = x @ self.E              # get embedding
            x = x.view(len(x), -1)      # stack the context
        h = F.relu(x @ self.W + self.b1)
        y = h @ self.U + self.b2
        # return F.softmax(y, dim=1)    # ne pas mettre softmax pour la Cross Entropy
        return y

    def parameters(self) -> List[torch.Tensor]:
        """ return model weights """
        if self.learn_embedding:
            return [self.W, self.b1, self.U, self.b2, self.E]
        else:
            return [self.W, self.b1, self.U, self.b2]
    
    def to(self, device: torch.device) -> None:
        """ move model weights to device """
        self.W.to(device)
        self.b1.to(device)
        self.U.to(device)
        self.b2.to(device)
        if self.learn_embedding:
            self.E.to(device)
    
    def save(self, path: str) -> None:
        """ save model weights in path/model.pt """
        path = os.path.join(path, 'model.pt')
        torch.save(self.parameters(), path)
    
    def load(self, path: str) -> None:
        """ load model weights """
        path += '_' + str(self.context_length) + '.pt'
        checkpoints = torch.load(path)
        self.W = checkpoints[0]
        self.b1 = checkpoints[1]
        self.U = checkpoints[2]
        self.b2 = checkpoints[3]
        if len(checkpoints) == 5:
            self.E = checkpoints[4]
        del checkpoints
    
    def copy_embedding(self, embedding: torch.Tensor) -> None:
        """ copy a matrix of embedding """
        self.E = embedding

    # def __str__(self) -> str:
    #     output = ''
    #     if self.learn_embedding:
    #         output += 'expedted input shape: (batch_size, 1)'
    #         output += 'Embedding layers'
    #     else:
    #         output += 'expedted input shape: (batch_size, ' + str(self.embedding_dim) + ')'
    #     output += '\n'
    #     output += 'FFN:'
    #     output += '\n'
    #     output += 'FFN:'
    #     return output


def get_model(config, embedding=None):
    model = Model(embedding_dim=config.model.embedding_dim, 
                  context_length=config.data.context_length, 
                  hidden_layer=config.model.hidden_layer, 
                  vocab_size=config.data.vocab_size,
                  learn_embedding=config.model.embedding.learn_embedding  )
    if embedding is not None:
        if config.model.embedding.learn_embedding and config.model.embedding.learn_from_vect_to_vect:
            model.copy_embedding(embedding)
            del embedding

    return model