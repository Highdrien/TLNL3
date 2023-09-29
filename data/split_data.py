import os
import random

random.seed(0)



def split(filetrain: str, new_train: str, new_val: str, split_rate: float) -> None:
    """ open filetrain and split it in 2 in order to have train and validation file"""

    assert 0 < split_rate < 1, 'split rate must be in ]0, 1['

    files_existing = os.listdir()
    assert not(new_train in files_existing and new_val in files_existing), 'train and validation data already exist'
        
    data_train = []
    data_val = []
    with open(file=filetrain, mode='r', encoding='utf8') as f:
        for line in f.readlines():
            if random.random() < split_rate:
                data_train.append(line)
            else:
                data_val.append(line)
    f.close()

    print(len(data_train))
    print(len(data_val))
    print('truth split rate:', len(data_train) / len(data_train + data_val))

    with open(file=new_train, mode='w', encoding='utf8') as f_train:
        for line in data_train:
            f_train.write(line)
    f_train.close()

    with open(file=new_val, mode='w', encoding='utf8') as f_val:
        for line in data_val:
            f_val.write(line)
    f_val.close()

    
if __name__ == '__main__':
    filetrain = 'Le_comte_de_Monte_Cristo.train.unk5.tok'

    new_train = 'Le_comte_de_Monte_Cristo.train.txt'
    new_val = 'Le_comte_de_Monte_Cristo.val.txt'
    split_rate = 0.8

    split(filetrain, new_train, new_val, split_rate)
