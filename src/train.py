import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from typing import Dict
import numpy as np

from src.data import DataGenerator, get_dataloader
from src.model import get_model
# import parameters as PARAM 
from src.metrics import compute_metrics
from config.utils import train_logger, train_step_logger
from utils.plot_learning_curves import save_learning_curves


def train(config: Dict) -> None:

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    # Get data
    train_generator = DataGenerator(config=config, mode='train')
    val_generator = DataGenerator(config=config, mode='val')
    
    embedding = None
    if config.model.embedding.learn_embedding and config.model.embedding.learn_from_vect_to_vect:
        embedding = train_generator.get_embedding()

    # Dataloader
    train_generator = get_dataloader(train_generator, config)
    val_generator = get_dataloader(val_generator, config)
    n_train, n_val = len(train_generator), len(val_generator)

    # Get model
    model = get_model(config, embedding)
    model.to(device)
    print(model)
    
    # Loss, optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=config.learning.milesstone, gamma=config.learning.gamma)

    logging_path = train_logger(config)
    best_val_loss = 10e6

    # Metrics
    metrics_name = list(filter(lambda x: config.metrics[x] is not None, config.metrics))

    ###############################################################
    # Start Training                                              #
    ###############################################################

    for epoch in range(1, config.learning.epochs + 1):
        print('epoch:', epoch)
        train_loss = 0
        train_range = tqdm(train_generator)
        train_metrics = np.zeros(len(metrics_name))

        # Training
        for x, y_true in train_range:
            
            x.to(device)
            y_true.to(device)

            y_pred = model.forward(x)

            loss = criterion(y_pred, y_true)

            train_loss += loss.item()
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)  
            train_metrics += compute_metrics(config, y_true, y_pred)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f" % (epoch, loss.item()))
            train_range.refresh()


        ###############################################################
        # Start Validation                                            #
        ###############################################################

        val_loss = 0
        val_range = tqdm(val_generator)
        val_metrics = np.zeros(len(metrics_name))

        with torch.no_grad():
            
            for x, y_true in val_range:
                x.to(device)
                y_true.to(device)

                y_pred = model.forward(x)

                loss = criterion(y_pred, y_true)
                
                val_loss += loss.item()
                y_pred = torch.nn.functional.softmax(y_pred, dim=1)  
                val_metrics += compute_metrics(config, y_true, y_pred)

                val_range.set_description("VAL -> epoch: %4d || loss: %4.4f" % (epoch, loss.item()))
                val_range.refresh()
        
        scheduler.step()

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        train_loss = train_loss / n_train
        val_loss = val_loss / n_val
        train_metrics = train_metrics / n_train
        val_metrics = val_metrics / n_train
        train_step_logger(logging_path, epoch, train_loss, val_loss, train_metrics, val_metrics)
        

        if val_loss < best_val_loss:
            print('save model weights')
            model.save(logging_path)
            best_val_loss = val_loss

    if config.learning.save_learning_curves:
        save_learning_curves(logging_path)
