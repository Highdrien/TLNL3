from typing import List

import torch
import torch.nn.functional as F

from src.data import DataGenerator, text_to_indexes
from src.model import get_model
# import parameters as PARAM 


def genere(config) -> List[int]:
    """ generate a text from an input text
    if the input text is None, generate a text from <s><s>...<s> """

    # Get vocab
    train_generator = DataGenerator(config=config, mode='train')
    
    vocab_size = train_generator.get_vocab_size()
    vocab = train_generator.vocab
    reversed_dico = train_generator.revese_dico()
    del train_generator

    if input is None:
        input = []
        output = []
    else:
        output = text_to_indexes(input, vocab.dico_voca)

    # Get model
    model = get_model(config, embedding=None)
    
    # Load model weigth
    model.load(PARAM.CHECKPOINT_PATH)

    # Complete output by <s> if output is too short
    if len(output) < PARAM.CONTEXT_LENGTH:
        s_index = vocab.get_word_index('<s>')
        for _ in range(PARAM.CONTEXT_LENGTH - len(output)):
            output.append(s_index)
        
    # Get context with embedding
    context = torch.zeros((PARAM.CONTEXT_LENGTH, PARAM.EMBEDDING_DIM))
    for k in range(PARAM.CONTEXT_LENGTH):
        context[k] = vocab.get_emb_torch(output[- PARAM.CONTEXT_LENGTH + k])
    context = context.reshape(PARAM.CONTEXT_LENGTH * PARAM.EMBEDDING_DIM)

    # remove possibility to generate <unk>
    index_to_ignore = vocab.get_word_index('<unk>')
    
    with torch.no_grad():
        nb_mots_predit = 0
        while nb_mots_predit < nb_mots and output[-1] != '</s>':
            new_word = model.forward(context)[0]
            new_word[index_to_ignore] = float('-inf')
            new_word = F.softmax(new_word, dim=0)

            idx = torch.argmax(new_word)
            output.append(reversed_dico[idx.item()])
            nb_mots_predit += 1

            context = torch.cat((context[PARAM.EMBEDDING_DIM:], vocab.get_emb_torch(idx)), dim=0)

    output = input + output[len(input):]
    return output            




# if __name__ == '__main__':
#     input = ['il', 'va', 'au', 'travail', 'sans']
#     print(input)
#     output = genere(nb_mots=10, input=input)
#     print(output)


    

