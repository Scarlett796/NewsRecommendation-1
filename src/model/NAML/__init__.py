import torch
from model.NAML.news_encoder import NewsEncoder
from model.NAML.user_encoder import UserEncoder
from model.NAML.click_predictor import ClickPredictor

def make_model(args, word_embedding):
    return NAML(args, word_embedding)

class NAML(torch.nn.Module):

    def __init__(self, args, word_embedding):
        super(NAML, self).__init__()
        self.args = args
        self.news_encoder = NewsEncoder(args, word_embedding)
        self.user_encoder = UserEncoder(args)
        self.click_predictor = ClickPredictor()
        self.news_cache = None

    def forward(self, batch):
        if batch['type'] == 'test':
            browsed_vector = []
            for news in batch['browsed']:
                if news['news_id'][0] in self.news_cache:
                    browsed_vector.append(self.news_cache[news['news_id'][0]])
                else:
                    tmp = self.news_encoder(news)
                    browsed_vector.append(tmp)
                    self.news_cache[news['news_id'][0]] = tmp
            browsed_vector = torch.stack(browsed_vector, dim=1).to(self.args.device)
            candidate_vector = []
            for news in batch['candidate']:
                if news['news_id'][0] in self.news_cache:
                    candidate_vector.append(self.news_cache[news['news_id'][0]])
                else:
                    tmp = self.news_encoder(news)
                    candidate_vector.append(tmp)
                    self.news_cache[news['news_id'][0]] = tmp
            candidate_vector = torch.stack(candidate_vector).to(self.args.device)
        else:
            # K+1, batch_size, n_filters
            candidate_vector = torch.stack([self.news_encoder(x) for x in batch['candidate']]).to(self.args.device)
            browsed_vector = torch.stack([self.news_encoder(x) for x in batch['browsed']], dim=1).to(self.args.device)
        # batch_size, n_filters
        user_vector = self.user_encoder(browsed_vector)
        predict = torch.stack([self.click_predictor(news_vector, user_vector) for news_vector in candidate_vector], dim=1)
        return predict
