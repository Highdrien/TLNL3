import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import numpy as np

from data import DataGenerator
from model import Model
import parameters as PARAM 


def train(save_weigth: bool) -> None:

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    # Get data
    train_generator = DataGenerator(mode='train',
                                    data_path=PARAM.DATA_PATH,
                                    context_length=PARAM.CONTEXT_LENGTH,
                                    embedding_dim=PARAM.EMBEDDING_DIM,
                                    line_by_line=True)
    
    vocab_size = train_generator.get_vocab_size()
    if vocab_size != PARAM.VOCAB_SIZE:
        print('attention: vocab_size != PARAM.VOCAB_SIZE')
        print(vocab_size)
    print('dataset size:', len(train_generator))

    train_generator = DataLoader(train_generator,
                                 batch_size=PARAM.BATCH_SIZE,
                                 shuffle=True,
                                 drop_last=False)

    # Get model
    model = Model(embedding_dim=PARAM.EMBEDDING_DIM, 
                  context_length=PARAM.CONTEXT_LENGTH, 
                  hidden_layer=PARAM.HIDDEN_LAYER, 
                  vocab_size=vocab_size)
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAM.LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=PARAM.MILESSTONE, gamma=PARAM.GAMMA)

    for epoch in range(1, PARAM.NUM_EPOCHS + 1):
        print('epoch:', epoch)
        total_loss = 0
        train_range = tqdm(train_generator)
        for x, y_true in train_range:
            
            x.to(device)
            y_true.to(device)

            y_pred = model.forward(x)

            loss = criterion(y_pred, y_true)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f" % (epoch, loss.item()))
            train_range.refresh()
        
        print('loss:', total_loss / len(train_generator))
        if save_weigth:
            model.save(PARAM.CHECKPOINT_PATH)
        scheduler.step()



if __name__ == '__main__':
    train(save_weigth=False)