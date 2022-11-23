import torch
from torch import nn as nn

from transformers import AutoModel, AutoTokenizer

def getBert(bert_name):
    bert = AutoModel.from_pretrained(bert_name)
    return bert

def getTokenizer(bert_name):
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    return tokenizer

class PretrainedModel(nn.Module):  

    def __init__(self, bert):
        super(PretrainedModel, self).__init__()
        self.bert = getBert(bert)
        self.linear = nn.Linear(self.bert.config.hidden_size, 128, bias=False)

    def forward(self, query_input_ids, query_token_type_ids, query_attention_mask, document_input_ids, document_token_type_ids, document_attention_mask):

        return self.score(self.query(query_input_ids, query_attention_mask), self.doc(document_input_ids, document_attention_mask)).reshape(-1, 1)
    
    def query(self, input_ids, attention_mask):
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask):
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids)).unsqueeze(2).cuda()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        return D

    def score(self, Q, D):
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[x != 1 for x in d] for d in input_ids.cpu().tolist()]
        return mask