from typing import List, Union, Dict
import os

from Vocab import Vocab


def get_vocab() -> Vocab:
    """ retourne le vocal du ficher embeddings"""
    embedding_path = os.path.join('data', 'embeddings-word2vecofficial.train.unk5.txt')
    return Vocab(embedding_path)


def openfile(file: str, line_by_line: bool=False) -> Union[List[str], List[List[str]]]:
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


def text_to_indexes(text: List[str], dico: Dict[str, int]) -> List[int]:
    """ transforme un textes de str en une liste d'indice des mots """
    new_list = []
    for word in text:
        if word in dico:
            new_list.append(dico[word])
        else:
            new_list.append('<unk>')
    return new_list


def split_text(text: List[int], k: int) -> List[List[int]]:
    """  
    fonction produit en sortie une liste dont les eléments sont des listes de k mots 
    consécutifs de T représentés par leur indices.
    """
    new_list = []
    for i in range(len(text) - k):
        new_list.append(text[i : i+k])
    return new_list


def process_data(file: str, vocab: Vocab, k: int) -> List[List[int]]:
    text = openfile(file)
    text = text_to_indexes(text, vocab.dico_voca)
    text = split_text(text, k + 1)
    return text


if __name__ == '__main__':
    file = os.path.join('data', 'Le_comte_de_Monte_Cristo.train.100.unk5.tok')
    vocab = get_vocab()
    text = process_data(file, vocab, 5)
    print(text)