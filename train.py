import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DataGenerator
from model import Model
import parameters as PARAM 


def train() -> None:

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    # Get data
    train_generator = DataGenerator(file=PARAM.FILE_TRAIN_1,
                                    context_length=PARAM.CONTEXT_LENGTH,
                                    embedding_dim=PARAM.EMBEDDING_DIM,
                                    line_by_line=True)
    
    vocab_size = train_generator.get_vocab_size()
    if vocab_size != PARAM.VOCAB_SIZE:
        print('attention: vocab_size != PARAM.VOCAB_SIZE')
        print(vocab_size)

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

    print('dataset size:', len(train_generator))

    for epoch in range(1, PARAM.NUM_EPOCHS + 1):
        print('epoch:', epoch)
        total_loss = 0

        for x, y_true in tqdm(train_generator):
            x.to(device)
            y_true.to(device)

            y_pred = model.forward(x)

            loss = criterion(y_pred, y_true)

            total_loss += loss.item()
            # print('loss:', loss.item(), end='\r')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print('loss:', total_loss / len(train_generator))
        model.save(PARAM.CHECKPOINT_PATH)



if __name__ == '__main__':
    train()