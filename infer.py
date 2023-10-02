import torch
# import torch.nn.functional as F
import torch.optim as optim

from data import create_generator
from model import Model
import parameters as PARAM 


def infer(x: torch.Tensor) -> torch.Tensor:

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    x.to(device)

    # Get model
    model = Model(embedding_dim=PARAM.EMBEDDING_DIM, 
                  context_length=PARAM.CONTEXT_LENGTH, 
                  hidden_layer=PARAM.HIDDEN_LAYER, 
                  vocab_size=PARAM.VOCAB_SIZE)
    model.to(device)

    # Load model weigth
    model.load(PARAM.CHECKPOINT_PATH)

    with torch.no_grad():
        y_pred = model.forward(x)
        return y_pred



if __name__ == '__main__':
    x = torch.zeros((PARAM.BATCH_SIZE, PARAM.EMBEDDING_DIM * PARAM.CONTEXT_LENGTH))
    y = infer(x)