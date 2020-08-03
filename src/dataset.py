from torch.utils.data import Dataset
import os
import pandas as pd
from ast import literal_eval

class MyDataset(Dataset):
    def __init__(self, args, type):
        super(MyDataset, self).__init__()
        self.args = args
        self.type = type
        self.empty_news = {
            'category': 0,
            'subcategory': 0,
            'title': [0 for _ in range(args.n_words_title)],
            'abstract': [0 for _ in range(args.n_words_abstract)]
        }

        if type == 'train':
            self.behaviors = pd.read_table(os.path.join(args.data_dir, 'train_behaviors.csv'), na_filter=False)
            self.news = pd.read_table(os.path.join(args.data_dir, 'train_news.csv'), index_col='news_id', na_filter=False)
        elif type == 'test':
            self.behaviors = pd.read_table(os.path.join(args.data_dir, 'test_behaviors.csv'), na_filter=False)
            self.news = pd.read_table(os.path.join(args.data_dir, 'test_news.csv'), index_col='news_id', na_filter=False)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, index):
        def get_news_dict(news_id):
            if news_id in self.news.index:
                row = self.news.loc[news_id]
                news = {
                    'category': row.category,
                    'subcategory': row.subcategory,
                    'title': literal_eval(row.title),
                    'abstract': literal_eval(row.abstract)
                }
            else:
                news = self.empty_news
            return news

        item = {}
        row = self.behaviors.iloc[index]
        if not row.history:
            tmp = []
            length = 0
        else:
            tmp = row.history.split(' ')
            length = len(tmp)
        if length < self.args.n_browsed_news:
            repeat_times = self.args.n_browsed_news - length
            item['browsed'] = [get_news_dict(news_id) for news_id in tmp]
            item['browsed'] += [self.empty_news for _ in range(repeat_times)]
        else:
            item['browsed'] = [get_news_dict(news_id) for news_id in tmp[:self.args.n_browsed_news]]
        tmp = row.impressions.split(' ')
        item['candidate'] = [get_news_dict(news_id) for news_id in tmp]
        if self.type == 'test':
            tmp = row.y_true.split(' ')
            item['y_true'] = [int(x) for x in tmp]
        return item