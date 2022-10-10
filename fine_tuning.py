## Importing the Libraries and loading the dataset ##

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, EarlyStoppingCallback
from huggingface_hub import notebook_login, HfFolder, HfApi
from collections import Counter
import evaluate
import numpy as np
import torch

## Using SST-2 dataset ##
raw_datasets = load_dataset("glue","sst2")

## Checking if GPU is available ##
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Name for the repository on the huggingface hub #
repo_name = "bert-base-sst2"

## Model used for fine-tuning ##
checkpoint = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

## Tokenization ##
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

## Data Pre-processing ##
tokenized_datasets = tokenized_datasets.remove_columns(['sentence','idx']) ## removing unwanted columns ##
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

## Create label2id, id2label dicts - to store id and label values ##
labels = tokenized_datasets["train"].features["labels"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    
### Training the Model ###
training_args = TrainingArguments(checkpoint)
training_args

### Training Arguments ###
training_args = TrainingArguments(
    output_dir=repo_name,
    num_train_epochs=15, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    learning_rate=5e-5,
    seed=33,
    # logging & evaluation strategies #
    logging_dir=f"{repo_name}/logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="tensorboard",
    # push to hub parameters #
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=repo_name,
    hub_token=HfFolder.get_token(),
    )
