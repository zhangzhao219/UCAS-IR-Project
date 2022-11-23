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
        self._classification_layer = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:,0,:]
        return self._classification_layer(output)
