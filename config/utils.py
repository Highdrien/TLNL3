import os
from numpy import exp
from typing import Dict, List
from datetime import datetime
from easydict import EasyDict as edict


def number_folder(path: str, name: str) -> str:
    """
    finds a declination of a folder name so that the name is not already taken
    """
    elements = os.listdir(path)
    last_index = -1
    for i in range(len(elements)):
        folder_name = name + str(i)
        if folder_name in elements:
            last_index = i
    return name + str(last_index + 1)


def train_logger(config: Dict) -> str:
    """
    creates a logs folder where we can find the config in confing.yaml and
    create train_log.csv which will contain the loss and metrics values
    """
    path = 'logs'
    if not os.path.exists(path):
        os.makedirs(path)
    folder_name = number_folder(path, config.name + '_')
    path = os.path.join(path, folder_name)
    os.mkdir(path)
    print(f'{path = }')

    # create train_log.csv where save the metrics
    with open(os.path.join(path, 'train_log.csv'), 'w') as f:
        first_line = 'step,' + config.learning.loss + ',val ' + config.learning.loss
        for metric in list(filter(lambda x: config.metrics[x], config.metrics)):
            first_line += ',' + metric
            first_line += ',val ' + metric
        f.write(first_line + '\n')
    f.close()

    # copy the config
    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        f.write("config_metadata: 'Saving time : " + date_time + "'\n")
        for line in config_to_yaml(config):
            f.write(line + '\n')
    f.close()

    return path


def config_to_yaml(config: Dict, space: str='') -> str:
    """
    transforms a dictionary (config) into a yaml line sequence
    """
    intent = ' ' * 4
    config_str = []
    for key, value in config.items():
        if type(value) == edict:
            if len(space) == 0:
                config_str.append('')
                config_str.append(space + '# ' + key + ' options')
            config_str.append(space + key + ':')
            config_str += config_to_yaml(value, space=space + intent)
        elif type(value) == str:
            config_str.append(space + key + ": '" + str(value) + "'")
        elif value is None:
            config_str.append(space + key + ": null")
        elif type(value) == bool:
            config_str.append(space + key + ": " + str(value).lower())
        else:
            config_str.append(space + key + ": " + str(value))
    return config_str


def train_step_logger(path: str,
                      epoch: int, 
                      train_loss: float,
                      val_loss: float, 
                      train_metrics: List[float], 
                      val_metrics: List[float]) -> None:
    """
    writes loss and metrics values in the train_log.csv
    """
    with open(os.path.join(path, 'train_log.csv'), 'a') as file:
        line = str(epoch) + ',' + str(train_loss) + ',' + str(val_loss)
        for i in range(len(train_metrics) - 1):
            line += ',' + str(train_metrics[i])
            line += ',' + str(val_metrics[i])
        # exp for the perplexity
        line += ',' + str(exp(train_metrics[-1]))
        line += ',' + str(exp(val_metrics[-1]))
        file.write(line + '\n')
    file.close()


def test_logger(path: str, metrics: List[str], values: List[float]) -> None:
    """
    creates a file 'test_log.txt' in the path containing for each line: metrics[i]: values[i]
    """
    with open(os.path.join(path, 'test_log.txt'), 'a') as f:
        for i in range(len(metrics) - 1):
            f.write(metrics[i] + ': ' + str(values[i]) + '\n')
        # exp for the perplexity
        f.write(metrics[-1] + ': ' + str(exp(values[-1])) + '\n')
