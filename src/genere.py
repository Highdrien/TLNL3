import os
from typing import List, Dict

import torch
import torch.nn.functional as F

from src.data import DataGenerator, text_to_indexes
from src.model import get_model


def genere(config: Dict, logging_path: str) -> List[int]:
    """ generate a text from an input text
    if the input text is None, generate a text from <s><s>...<s> """

    context_length = config.data.context_length
    vocab_size = config.data.vocab_size

    # Get vocab
    train_generator = DataGenerator(config=config, mode='train')
    
    vocab = train_generator.vocab
    reversed_dico = train_generator.revese_dico()
    del train_generator

    # get input data
    input = get_input(config.generate.folder, config.generate.input_file)
    print('input:', input)
    if input is None:
        input = []
        output = []
    else:
        output = text_to_indexes(input, vocab.dico_voca)

    # Get model and load weigth
    model = get_model(config)
    model.load(os.path.join(logging_path, 'model.pt'))
    print(model)

    # Complete output by <s> if output is too short
    if len(output) < context_length:
        s_index = vocab.get_word_index('<s>')
        for _ in range(context_length - len(output)):
            output.append(s_index)

    # Get context
    if not config.model.embedding.learn_embedding:
        # Get context with embedding
        context = torch.zeros((context_length, config.model.embedding_dim))
        for k in range(context_length):
            context[k] = vocab.get_emb_torch(output[- context_length + k])
        context = context.reshape(context_length * config.model.embedding_dim)
        context = context.unsqueeze(0)
    else:
        # Get context with one-hot encoding of indexes
        context = torch.tensor(output[-context_length:])
        context = F.one_hot(context, num_classes=vocab_size).unsqueeze(0)
        context = context.type(torch.float32)

    # remove possibility to generate <unk>
    index_to_ignore = vocab.get_word_index('<unk>')
    
    # Start generation
    with torch.no_grad():
        nb_mots_predit = 0
        while nb_mots_predit < config.generate.nb_words and output[-1] != '</s>':
            new_word = model.forward(context)[0]
            new_word[index_to_ignore] = float('-inf')
            new_word = F.softmax(new_word, dim=0)

            idx = torch.argmax(new_word)
            output.append(reversed_dico[idx.item()])
            nb_mots_predit += 1

            if not config.model.embedding.learn_embedding:
                context = context.squeeze(0)
                context = torch.cat((context[config.model.embedding_dim:], vocab.get_emb_torch(idx)), dim=0)
                context = context.unsqueeze(0)
            else:  
                new_word = F.one_hot(idx, num_classes=vocab_size).unsqueeze(0).unsqueeze(0)
                context = torch.cat((context[:, -4:, :], new_word), dim=1)

    output = input + output[len(input):]
    print('output:', output)
    write_output(generate_path=config.generate.folder,
                 output=output,
                 output_file=config.generate.output_file,
                 logging_path=logging_path)        


def get_input(generate_path: str, input_file: str) -> List[str]:
    if input_file not in os.listdir(generate_path):
        return None
    input = []
    with open(os.path.join(generate_path, input_file), mode='r', encoding='utf8') as f:
        for line in f.readlines():
            input += line.split(' ')
        f.close()
    input = [word for word in input if word != '']
    return input


def write_output(generate_path: str, output: List[str], output_file: str, logging_path: str) -> None:
    if '<experiment_name>' in output_file:
        split = output_file.split('<experiment_name>')
        output_file = split[0] + logging_path.split('\\')[-2] + split[1]
        print(output_file)
    output_path = os.path.join(generate_path, output_file)
    with open(output_path, mode='w', encoding='utf8') as f:
        for word in output:
            f.write(str(word) + ' ')
        f.write('\n')
        f.close()
