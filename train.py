import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from data import DataGenerator
from model import Model
import parameters as PARAM 
from metrics import accuracy


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
    val_generator = DataGenerator(mode='val',
                                  data_path=PARAM.DATA_PATH,
                                  context_length=PARAM.CONTEXT_LENGTH,
                                  embedding_dim=PARAM.EMBEDDING_DIM,
                                  line_by_line=True)
    
    vocab_size = train_generator.get_vocab_size()
    assert vocab_size == PARAM.VOCAB_SIZE, 'Warning: vocab_size != PARAM.VOCAB_SIZE'
    print('dataset size:', len(train_generator))

    train_generator = DataLoader(train_generator,
                                 batch_size=PARAM.BATCH_SIZE,
                                 shuffle=True,
                                 drop_last=False)
    val_generator = DataLoader(val_generator,
                               batch_size=PARAM.BATCH_SIZE,
                               shuffle=True,
                               drop_last=False)

    # Get model
    model = Model(embedding_dim=PARAM.EMBEDDING_DIM, 
                  context_length=PARAM.CONTEXT_LENGTH, 
                  hidden_layer=PARAM.HIDDEN_LAYER, 
                  vocab_size=vocab_size)
    model.to(device)
    
    # Loss, optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAM.LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=PARAM.MILESSTONE, gamma=PARAM.GAMMA)

    n_train, n_val = len(train_generator), len(val_generator)

    for epoch in range(1, PARAM.NUM_EPOCHS + 1):
        print('epoch:', epoch)
        train_loss = 0
        train_acc = 0
        train_range = tqdm(train_generator)

        # Training
        for x, y_true in train_range:
            
            x.to(device)
            y_true.to(device)

            y_pred = model.forward(x)

            loss = criterion(y_pred, y_true)

            train_loss += loss.item()
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)  
            train_acc += accuracy(y_true, y_pred)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f" % (epoch, loss.item()))
            train_range.refresh()

        print("TRAIN -> loss: %4.4f | acc: %4.2f" % (train_loss / n_train, train_acc / n_train))

        
        # Validation
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            val_range = tqdm(val_generator)
            for x, y_true in val_range:
                x.to(device)
                y_true.to(device)

                y_pred = model.forward(x)

                loss = criterion(y_pred, y_true)
                
                val_loss += loss.item()
                y_pred = torch.nn.functional.softmax(y_pred, dim=1)  
                val_acc += accuracy(y_true, y_pred)

                val_range.set_description("VAL -> epoch: %4d || loss: %4.4f" % (epoch, loss.item()))
                val_range.refresh()
        

        print("VAL -> loss: %4.4f | acc: %4.2f" % (val_loss / n_val, val_acc / n_val))

        if save_weigth:
            model.save(PARAM.CHECKPOINT_PATH)
        scheduler.step()



if __name__ == '__main__':
    train(save_weigth=False)