from typing import List, Dict
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
        self.hidden_layer = hidden_layer

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
    
    def __str__(self) -> str:
        """ return model info """
        output = '\n----------\nB - batch size \n'
        output += 'K=' + str(self.context_length) + ' - context length \n'
        output += 'V=' + str(self.vocab_size) + ' - Vocab size\n'
        output += 'HL=' + str(self.hidden_layer) + ' - hidden layers \n'

        if self.learn_embedding:
            output += 'E=' + str(self.embedding_dim) + ' - Embedding dimension \n'
            output += '---\n'
            output += 'input shape: (B, K, V) \n'
            output += 'Embedding -> output shape: (B, K, E)  -> param:' + str(self.vocab_size * self.embedding_dim)+' \n'
            output += 'Stack     -> output shape: (B, K * E) -> param:0 \n'
        
        else:
            output += '---\n'
        
        output += 'FFN1 relu -> output shape: (B, HL)    -> param:' + str((self.embedding_dim * self.context_length + 1) * self.hidden_layer) + '\n'
        output += 'FF1       -> output shape: (B, V)     -> param:' + str((self.hidden_layer + 1) * self.vocab_size) + '\n'
        output += '---\n'

        output += 'number of parameters: ' + str(self.number_parameters())
        output += '\n----------\n'
        return output

    def number_parameters(self) -> int:
        """ get the number of parameters of the model """
        params = 0
        if self.learn_embedding:
            params += self.vocab_size * self.embedding_dim
        params += (self.embedding_dim * self.context_length + 1) * self.hidden_layer
        params += (self.hidden_layer + 1) * self.vocab_size
        return params


def get_model(config: Dict, embedding: torch.Tensor=None):
    """ get the model according the configuration 
        if the model have to learn his own embedding from vect2vect, the model will copy the embedding"""
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