import os
from typing import List, Union, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from Vocab import Vocab

# Type: # text can be a list of words or a list of sentences (ie list of list of words)
Text_type   = Union[List[str], List[List[str]]]
Indix_Type  = Union[List[int], List[List[int]]]


def openfile(file: str, line_by_line: bool=False) -> Text_type:
    """
    take a file name and return a list of the word of the file
    if line_by_line=True, the output will be a list of sentences
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
    def __init__(self, 
                 mode: str,
                 data_path: str, 
                 context_length: int, 
                 embedding_dim: int, 
                 line_by_line: bool) -> None:

        assert mode in ['train', 'val', 'test'], "mode must be 'train', 'val', or 'test'"
        print('mode:', mode)
        file = os.path.join(data_path, 'Le_comte_de_Monte_Cristo.' + mode + '.txt')

        self.vocab = self.get_vocab()
        self.vocab_size = len(self.vocab.dico_voca)
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.data_path = data_path

        text = openfile(file, line_by_line)
        text = text_to_indexes(text, self.vocab.dico_voca)
        self.data = self.split_text(text)

        if line_by_line:
            new_data = []
            for sentence in self.data:
                new_data += sentence
            self.data = new_data


    def __len__(self) -> int:
        """ return the number of data """
        return len(self.data)
    

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        take a data index and return x, y such that:
        - x are a tensor with a shape: (context_length, embedding_dim) which is the context of the sentence
        - y are a hot-one encoding tensor which represents the index of the predicted word
        """
        x = torch.zeros((self.context_length, self.embedding_dim))
        for i in range(self.context_length):
            x[i] = self.vocab.get_emb_torch(self.data[index][i])
        x = x.view(self.context_length * self.embedding_dim)
        y = torch.nn.functional.one_hot(torch.tensor(self.data[index][-1]), num_classes=self.vocab_size)
        y = y.type(torch.float32)
        return x, y
    

    def get_vocab(self) -> Vocab:
        """ return the Vocal associated with embedding_file """
        embedding_path = os.path.join('data', 'embeddings-word2vecofficial.train.unk5.txt')
        return Vocab(embedding_path)
    

    def get_vocab_size(self) -> int:
        """ return size of the dictionary of the Vocab """
        return self.vocab_size
    

    def split_text(self, text: Indix_Type) -> List[Indix_Type]:
        """ take a text and return all the <context_length> consecutive words """
        new_list = []
        if type(text[0]) == int:
            # text is a list of word index
            for i in range(len(text) - self.context_length):
                new_list.append(text[i : i + self.context_length + 1])

        else:
            # text is a list of sentences
            for sentence in text:
                new_sentence = []
                for i in range(len(sentence) - self.context_length):
                    new_sentence.append(sentence[i : i + self.context_length + 1])
                new_list.append(new_sentence)
        
        return new_list
    
    def revese_dico(self) -> List[str]:
        """ return the revesed dictionary
        the output is a list such that  if dict[word] = i also output[i] = word"""
        reversed_dico = [0 for i in range(self.vocab_size)]
        for key, value in self.vocab.dico_voca.items():
            reversed_dico[value] = key
        return reversed_dico

    


def text_to_indexes(text: Text_type, 
                    dico: Dict[str, int]) -> Indix_Type:
    """ transforms str text into a word index list 
    if text is a word list, then new_list will be an index list
    if text is a phrase list (= word list then new_list will be an index list)
    when a word in text is not in the dictionary, it is replaced by: '<unk>'
    """
    new_list = []
    if type(text[0]) == str:
        # text is a list of words
        for word in text:
            if word in dico:
                new_list.append(dico[word])
            else:
                new_list.append('<unk>')

    else:
        # text is a list of sentences
        for sentence in text:
            new_sentence = []
            for word in sentence:
                if word in dico:
                    new_sentence.append(dico[word])
                else:
                    new_sentence.append('<unk>')
            new_list.append(new_sentence)
    
    return new_list


if __name__ == '__main__':
    dataset = DataGenerator('train', data_path='data', context_length=3, embedding_dim=100, line_by_line=True)
    x, y = dataset.__getitem__(3)
    print('without DATALOADER')
    print(x.shape)
    print(y.shape)
    print()

    data_generator = DataLoader(dataset,batch_size=10, shuffle=True, drop_last=True)
    print('with DATALOADER')
    for x, y in data_generator:
        print(x.shape)
        print(y.shape)
        exit()