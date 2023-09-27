from typing import List, Union, Dict
import os
import torch

from Vocab import Vocab

# Type: # les textes pouvent être des liste de mots où des liste des phrases.
Text_type   = Union[List[str], List[List[str]]]
Indix_Type  = Union[List[int], List[List[int]]]


def get_vocab() -> Vocab:
    """ retourne le vocal du ficher embeddings"""
    embedding_path = os.path.join('data', 'embeddings-word2vecofficial.train.unk5.txt')
    return Vocab(embedding_path)


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


def text_to_indexes(text: Text_type, 
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



def split_text(text: Indix_Type, k: int) -> List[Indix_Type]:
    """  
    fonction produit en sortie une liste dont les eléments sont des listes 
    de k mots consécutifs de T représentés par leur indices.
    """
    new_list = []
    if type(text[0]) == int:
        # text est une liste d'indices
        for i in range(len(text) - k):
            new_list.append(text[i : i + k])

    else:
        # text est une liste de liste d'indices
        for sentence in text:
            new_sentence = []
            for i in range(len(sentence) - k):
                new_sentence.append(sentence[i : i + k])
            new_list.append(new_sentence)
    
    return new_list


def process_data(file: str, vocab: Vocab, k: int, line_by_line: bool=False) -> List[Indix_Type]:
    """ 
    prend un filcher texte, un vocabulaire, une longueur de contexte et 
    renvoie un ensemble de liste d'indice de longueur k + 1 (k mots du contexte et le mot à prédire)
    Vous pouvez choisir de récupérer le texte avec une liste de mot ou une liste de phrase avec le parametre: line_by_line
    si line_by_line=False -> type de sortie: List[List[int]]
    sinon                 -> type de sortie: List[List[List[int]]]
    """
    text = openfile(file, line_by_line)
    text = text_to_indexes(text, vocab.dico_voca)
    text = split_text(text, k + 1)
    return text


def split_context(data: List[int], vocab: Vocab, k: int, num_classes: int, embedding_dim: int):
    """ 
    prend un contexte de longueur k+1 et renvoie x, y tq:
        - x est une matrice k,embedding_dim contenant les embeddings des k premiers mot
        - y est un tensor hot-one qui corrspond à l'indice du mots à prédire
    """
    x = torch.zeros((k, embedding_dim))
    for i in range(len(data) - 1):
        x[i] = vocab.get_emb_torch(data[i])
    x = x.view(k * embedding_dim)
    y = torch.nn.functional.one_hot(torch.tensor(data[-1]), num_classes=num_classes)
    y = y.type(torch.float32)
    return x, y




if __name__ == '__main__':
    file = os.path.join('data', 'Le_comte_de_Monte_Cristo.train.100.unk5.tok')
    vocab = get_vocab()
    text = process_data(file, vocab, 5, line_by_line=True)
    print(text)