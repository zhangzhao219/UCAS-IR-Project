import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def getBert(bert_name):
    bert = AutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=1, ignore_mismatched_sizes=True)
    return bert

def getTokenizer(bert_name):
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    return tokenizer

class PretrainedModel(nn.Module):
    def __init__(self, bert):
        super(PretrainedModel, self).__init__()
        self.bert = getBert(bert)

    def forward(self,input_ids,token_type_ids,attention_mask):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return output.logits