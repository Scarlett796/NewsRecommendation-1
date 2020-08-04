import torch
from option import args
from preprocess import preprpocess_data
from loader import Loader
from model import Model
from trainer import Trainer
from cache import NewsCache
import numpy as np
import os

torch.manual_seed(args.seed)

def main():
    if args.pre:
        preprpocess_data(args)
    else:
        print('Skip data preprocessing')
    try:
        word_embedding = torch.from_numpy(
            np.load(os.path.join(args.data_dir, 'word_embedding.npy'))).float()
    except FileNotFoundError:
        word_embedding = None
    my_model = Model(args, word_embedding)
    my_loader = Loader(args)
    NewsCache(args, my_model, False)
    my_trainer = Trainer(args, my_model, my_loader)
    my_trainer.test()

    #while not my_trainer.terminate():
    #    my_trainer.train()
    #    my_trainer.test()
    #my_trainer.plot_loss()


if __name__ == '__main__':
    main()
