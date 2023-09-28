from os.path import join


# Data param
CONTEXT_LENGTH = 5
FILE_TRAIN_0 = join('data', 'Le_comte_de_Monte_Cristo.train.100.unk5.tok')
FILE_TRAIN_1 = join('data', 'Le_comte_de_Monte_Cristo.train.unk5.tok')

# Model param
EMBEDDING_DIM = 100
HIDDEN_LAYER = 100
VOCAB_SIZE = 2908

# Training param
NUM_EPOCHS = 10
LEARNING_RATE = 0.1
BATCH_SIZE = 64

CHECKPOINT_PATH = 'checkpoint'