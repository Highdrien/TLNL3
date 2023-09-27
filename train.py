import torch
import torch.nn.functional as F
import os
from typing import List
import random
import torch.optim as optim

from data import process_data, get_vocab
from Vocab import Vocab


def forward(x, W, b1, U, b2):
    h = F.relu(x @ W + b1)
    y = h @ U + b2
    return F.softmax(y, dim=1)


def split_data(data: List[int], vocab: Vocab, k: int, num_classes: int):
    x = torch.zeros((k, 100))
    for i in range(len(data) - 1):
        x[i] = vocab.get_emb_torch(data[i])
    x = x.view(k * 100)
    y = F.one_hot(torch.tensor(data[-1]), num_classes=num_classes)
    return x, y


def train():
    vocab = get_vocab()

    vocab_size = len(vocab.dico_voca)
    k = 3                   # Nombre de mots consécutifs
    d = 100                 # Dimension des plongements
    dh = 100                # Dimension de la couche cachée
    num_epochs = 1          
    learning_rate = 0.01

    file_train = os.path.join('data', 'Le_comte_de_Monte_Cristo.train.100.unk5.tok')
    text = process_data(file_train, vocab, k=k)

    W = torch.randn((d * k, dh), requires_grad=True)
    b1 = torch.randn((1, dh), requires_grad=True)
    U = torch.randn((dh, vocab_size), requires_grad=True)
    b2 = torch.randn((1, vocab_size), requires_grad=True)
    
    optimizer = optim.Adam([W, b1, U, b2], lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        print('epoch:', epoch)
        random.shuffle(text)
        for i in range(len(text)):
            data = text[i]
            x, y = split_data(data, vocab, k, num_classes=vocab_size)
            y_pred = forward(x, W, b1, U, b2)
            y_pred[0]
            loss = F.cross_entropy(y, y_pred[0])
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            print('loss:', loss.item())



        








if __name__ == '__main__':
    train()