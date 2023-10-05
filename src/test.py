import os
import torch
from tqdm import tqdm
from typing import Dict
import numpy as np

from src.data import DataGenerator, get_dataloader
from src.model import get_model
# import parameters as PARAM 
from src.metrics import compute_metrics
from config.utils import test_logger


def test(config: Dict, logging_path: str) -> None:

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    # Get data
    test_generator = DataGenerator(config=config, mode='test')

    # Dataloader
    test_generator = get_dataloader(test_generator, config)
    n_test = len(test_generator)

    # Get model
    model = get_model(config)
    model.to(device)
    model.load(os.path.join(logging_path, 'model.pt'))
    print(model)

    # Loss and Metrics
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    metrics_name = list(filter(lambda x: config.metrics[x] is not None, config.metrics))


    ###############################################################
    # Start Testing                                               #
    ###############################################################

    test_loss = 0
    test_metrics = np.zeros(len(metrics_name))

    with torch.no_grad():
        
        for x, y_true in tqdm(test_generator, desc='testing'):
            x.to(device)
            y_true.to(device)

            y_pred = model.forward(x)

            loss = criterion(y_pred, y_true)
            
            test_loss += loss.item()
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)  
            test_metrics += compute_metrics(config, y_true, y_pred)

    ###################################################################
    # Save Scores in logs                                             #
    ###################################################################

    metrics_name = [config.learning.loss] + metrics_name
    test_metrics = [test_loss / n_test] + list(test_metrics / n_test)
    print(metrics_name)
    print(test_metrics)

    
    test_logger(logging_path, metrics_name, test_metrics)
