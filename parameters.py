# Data param
CONTEXT_LENGTH = 5
DATA_PATH = 'data'

# Model param
EMBEDDING_DIM = 100
HIDDEN_LAYER = 1024
VOCAB_SIZE = 2908

# Training param
NUM_EPOCHS = 10
BATCH_SIZE = 128

# Learning rate and scheduler: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html
LEARNING_RATE = 0.1
MILESSTONE = [5]
GAMMA = 0.1

CHECKPOINT_PATH = 'checkpoint'