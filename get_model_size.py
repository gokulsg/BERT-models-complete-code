import torch
from transformers import AutoModelForSequenceClassification

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

checkpoint = 'gokuls/bert_uncased_L-12_H-768_A-12_emotion'
model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
param_value = '{:,}'.format(count_parameters(model))
print('Number of parameters: ', param_value)
