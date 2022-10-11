## Computing the model size and the processing time ##
## Importing the libraries ##
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

## Compute processing time ##
## Here, we are analyzing the performance on CPU ##

checkpoint = "gokuls/bert-base-Massive-intent" ## Model used for analysis - model from huggingface hub ##

## Tokenization ##
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

## Model ##
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

## Eample sentence ##
example = "wake me up at eight am on monday"

## Getting processing time ##
start_time = time.time()

inputs = tokenizer(example, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
end_time =time.time()

print('The processing time is: ', end_time-start_time)

## Getting the number of parameters ##
print('Number of Parameters: ', sum(p.numel() for p in model.parameters()))
