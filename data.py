import os
from typing import List, Union, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from Vocab import Vocab

# Type: # les textes pouvent être des liste de mots où des liste des phrases.
Text_type   = Union[List[str], List[List[str]]]
Indix_Type  = Union[List[int], List[List[int]]]


def openfile(file: str, line_by_line: bool=False) -> Text_type:
    """
    prend un chemin ou une liste de chemain vers un texte 
    et renvoie la liste de mots que contients ce texte
    si line_by_line = True, renvoie la liste de phrase (donc une list de list de mots)
    """
    text = []

    with open(file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.split(' ')
            if line[-1] == '\n':
                line = line[:-1]
            if line[0] == '<s>':
                line = line[1:]
            if line[0] == '<s>':
                line = line[1:]
            
            if not(line_by_line):
                text += line
            else:
                text.append(line)
    
    return text


class DataGenerator(Dataset):
    def __init__(self, file: str, context_length: int, embedding_dim: int, line_by_line: bool) -> None:
        self.vocab = self.get_vocab()
        self.vocab_size = len(self.vocab.dico_voca)
        self.context_length = context_length
        self.embedding_dim = embedding_dim

        text = openfile(file, line_by_line)
        text = self.text_to_indexes(text, self.vocab.dico_voca)
        self.data = self.split_text(text)

        if line_by_line:
            new_data = []
            for sentence in self.data:
                new_data += sentence
            self.data = new_data


    def __len__(self) -> int:
        """ renvoie le nombre de données """
        return len(self.data)
    

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        prend un index des données et renvoie x, y tq:
        - x est une matrice k,embedding_dim contenant les embeddings des k premiers mot
        - y est un tensor hot-one qui corrspond à l'indice du mots à prédire
        """
        x = torch.zeros((self.context_length, self.embedding_dim))
        for i in range(self.context_length):
            x[i] = self.vocab.get_emb_torch(self.data[index][i])
        x = x.view(self.context_length * self.embedding_dim)
        y = torch.nn.functional.one_hot(torch.tensor(self.data[index][-1]), num_classes=self.vocab_size)
        y = y.type(torch.float32)
        return x, y
    

    def get_vocab(self) -> Vocab:
        """ retourne le vocal du ficher embeddings"""
        embedding_path = os.path.join('data', 'embeddings-word2vecofficial.train.unk5.txt')
        return Vocab(embedding_path)
    

    def get_vocab_size(self) -> int:
        """ retourne la taille du vocabulaire """
        return self.vocab_size


    def text_to_indexes(self,
                        text: Text_type, 
                        dico: Dict[str, int]) -> Indix_Type:
        """ transforme un textes de str en une liste d'indice des mots 
        si text est une liste de mots alors new_list sera une liste d'index
        si text est une liste phrase (= liste de liste de mots alors new_list sera une liste de liste d'index)
        lorsqu'un mot du text n'est pas dans le dico, il est remplacé par: '<unk>'
        """
        new_list = []
        if type(text[0]) == str:
            # text est une liste de mots
            for word in text:
                if word in dico:
                    new_list.append(dico[word])
                else:
                    new_list.append('<unk>')

        else:
            # text est une liste de phrase
            for sentence in text:
                new_sentence = []
                for word in sentence:
                    if word in dico:
                        new_sentence.append(dico[word])
                    else:
                        new_sentence.append('<unk>')
                new_list.append(new_sentence)
        
        return new_list
    

    def split_text(self, text: Indix_Type) -> List[Indix_Type]:
        """  
        fonction produit en sortie une liste dont les eléments sont des listes 
        de <context_length> mots consécutifs de T représentés par leur indices.
        """
        new_list = []
        if type(text[0]) == int:
            # text est une liste d'indices
            for i in range(len(text) - self.context_length):
                new_list.append(text[i : i + self.context_length + 1])

        else:
            # text est une liste de liste d'indices
            for sentence in text:
                new_sentence = []
                for i in range(len(sentence) - self.context_length):
                    new_sentence.append(sentence[i : i + self.context_length + 1])
                new_list.append(new_sentence)
        
        return new_list

    

def create_generator(file: str, 
                     context_length: int, 
                     embedding_dim: int, 
                     line_by_line: bool, 
                     batch_size: int,
                     shuffle: bool,
                     drop_last: bool) -> Tuple[DataLoader, int]:
    """ Renvoie le dataloader et la taille du vocabulaire"""
    
    dataset = DataGenerator(file, 
                            context_length=context_length, 
                            embedding_dim=embedding_dim, 
                            line_by_line=line_by_line)

    generator = DataLoader(dataset,
                           batch_size=batch_size, 
                           shuffle=shuffle, 
                           drop_last=drop_last)
    
    return generator, dataset.get_vocab_size()


if __name__ == '__main__':
    file = os.path.join('data', 'Le_comte_de_Monte_Cristo.train.100.unk5.tok')
    dataset = DataGenerator(file, context_length=3, embedding_dim=100, line_by_line=True)
    x, y = dataset.__getitem__(3)
    print('sans le DATALOADER')
    print(x.shape)
    print(y.shape)
    print()

    data_generator = DataLoader(dataset,batch_size=10, shuffle=True, drop_last=True)
    print('Avec le DATALOADER')
    for x, y in data_generator:
        print(x.shape)
        print(y.shape)
        exit()