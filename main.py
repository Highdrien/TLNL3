import os
import yaml
import argparse
from easydict import EasyDict as edict

from src.train import train
from src.test import test
from src.genere import genere


def load_config(path='configs/config.yaml'):
    stream = open(path, 'r')
    return edict(yaml.safe_load(stream))


def find_config(experiment_path):
    yaml_in_path = list(filter(lambda x: x[-5:] == '.yaml', os.listdir(experiment_path)))

    if len(yaml_in_path) == 1:
        return os.path.join(experiment_path, yaml_in_path[0])

    if len(yaml_in_path) == 0:
        print("ERROR: config.yaml wasn't found in", experiment_path)
    
    if len(yaml_in_path) > 0:
        print("ERROR: a lot a .yaml was found in", experiment_path)
    
    exit()


def main(options):
    if options['mode'] == 'train':
        config = load_config(options['config_path'])
        train(config)
    elif options['mode'] in ['genere', 'generate']:
        config = find_config(options['path'])
        config = load_config(config)
        genere(config, options['path'], temperature=0.6, top_k=5)
    elif options['mode'] == 'test':
        config = find_config(options['path'])
        config = load_config(config)
        test(config, options['path'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str, help="choose a mode between 'train', 'genere'")
    parser.add_argument('--config_path', default=os.path.join('config', 'config.yaml'), type=str, help="path to config (for training)")
    parser.add_argument('--path', type=str, help="experiment path (for test, prediction or generate)")

    args = parser.parse_args()
    options = vars(args)

    main(options)