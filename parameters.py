from os.path import join


# Data param
CONTEXT_LENGTH = 5
FILE_TRAIN_0 = join('data', 'Le_comte_de_Monte_Cristo.train.100.unk5.tok')
FILE_TRAIN_1 = join('data', 'Le_comte_de_Monte_Cristo.train.unk5.tok')

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