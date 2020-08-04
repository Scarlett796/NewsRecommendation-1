from torch.utils.data import Dataset, DataLoader
import torch
import os
import pandas as pd
from ast import literal_eval
from tqdm import tqdm

class NewsDataset(Dataset):
    def __init__(self, args):
        super(NewsDataset, self).__init__()
        self.args = args
        self.news = pd.read_table(os.path.join(args.data_dir, 'train_news.csv'), na_filter=False)

    def __len__(self):
        return len(self.news)

    def __getitem__(self, index):
        row = self.news.iloc[index]
        item = {
            'news_id': row.news_id,
            'category':  (row.category),
            'subcategory': row.subcategory,
            'title': literal_eval(row.title),
            'abstract': literal_eval(row.abstract)
        }
        return item

class NewsLoader:
    def __init__(self, args):
        self.news_loader = DataLoader(
            NewsDataset(args),
            batch_size=1,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=not args.cpu
        )

class NewsCache:
    def __init__(self, args, model, load=False):
        self.news_loader = NewsLoader(args).news_loader
        self.news_cache = {}
        if load:
            print('Loading news vector cache from', os.path.join(args.data_dir, 'news_cache.pt'))
            self.news_cache = torch.load(os.path.join(args.data_dir, 'news_cache.pt'))
        else:
            with torch.no_grad():
                model.eval()
                with tqdm(total=len(self.news_loader), desc='Generating news vector cache') as p:
                    for i, news in enumerate(self.news_loader):
                        self.news_cache[news['news_id'][0]] = model.model.news_encoder(news)
                        p.update(1)
                torch.save(self.news_cache, os.path.join(args.data_dir, 'news_cache.pt'))
                print('News vector cache saved as', os.path.join(args.data_dir, 'news_cache.pt'))
        model.model.news_cache = self.news_cache