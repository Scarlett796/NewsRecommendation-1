import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import datetime
from metrics import AUC, MRR, nDCG
import numpy as np

class Trainer:
    def __init__(self, args, model, loader):
        self.args = args
        self.model = model
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr,
                                          betas=args.betas, eps=args.epsilon)
        self.loss_list = []
        self.epoch = 0
        self.batch_num = 0
        if not os.path.exists('../result'):
            os.mkdir('../result')
        if self.args.save is None:
            dir_name = self.args.model + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        else:
            dir_name = self.args.save
        self.save_dir = os.path.join('../result', dir_name)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.exists(os.path.join(self.save_dir, 'checkpoints')):
            os.mkdir(os.path.join(self.save_dir, 'checkpoints'))

    def train(self):
        self.model.train()
        print('[Epoch ' + str(self.epoch + 1) + ']')
        for i, batch in enumerate(self.train_loader):
            batch['type'] = 'train'
            self.optimizer.zero_grad()
            predict = self.model(batch)
            loss = torch.stack([x[0] for x in -F.log_softmax(predict, dim=1)]).mean()
            self.loss_list.append(loss.item())
            print('\tbatch_num:', i+1, '\tloss:', loss.item())
            loss.backward()
            self.optimizer.step()
            self.batch_num += 1
        self.epoch += 1
        self.save()

    def test(self):
        with torch.no_grad():
            self.model.eval()
            AUC_list, MRR_list, nDCG5_list, nDCG10_list = [], [], [], []
            with tqdm(total=len(self.test_loader), desc='Testing') as p:
                for i, batch in enumerate(self.test_loader):
                    batch['type'] = 'test'
                    y_true = torch.stack(batch['y_true']).squeeze(dim=1).tolist()
                    predict = self.model(batch).squeeze(dim=0).tolist()
                    AUC_list.append(AUC(y_true, predict))
                    MRR_list.append(MRR(y_true, predict))
                    nDCG5_list.append(nDCG(y_true, predict, 5))
                    nDCG10_list.append(nDCG(y_true, predict, 10))
                    p.update(1)
            print('AUC:', np.mean(AUC_list))
            print('MRR:', np.mean(MRR_list))
            print('nDCG@5:', np.mean(nDCG5_list))
            print('nDCG@10:', np.mean(nDCG10_list))

    def save(self):
        print('Saving model...')
        checkpoint = {
            'model': self.model.model.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoints', 'epoch' + str(self.epoch) + '.pt'))
        print('Model saved as', os.path.join(self.save_dir, 'checkpoints', 'epoch' + str(self.epoch) + '.pt'))

    def terminate(self):
        if not self.args.train:
            self.test()
            return True
        else:
            return self.epoch >= self.args.epochs

    def plot_loss(self):
        plt.plot([_ for _ in range(self.batch_num)], self.loss_list)
        plt.savefig(os.path.join(self.save_dir, 'loss.png'))