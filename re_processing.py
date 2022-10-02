import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer
import torch

f = open("train_ds")
sequence = f.read()

dataset = sequence.split("\"}{\"")


def get_training_corpus():

    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples


training_corpus = get_training_corpus()

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
new_tokenizer = tokenizer.train_new_from_iterator(training_corpus, 52000)

for data in dataset:
    new_tokens = new_tokenizer.tokenize(data)
    
    for token in new_tokens:
        if token == "C" or token == "H" or token == "E" or token == "\"" or token == "{" or token == "}":
            continue
        else:
            print(token)






from torch.utils.data import DataLoader

train_dataloader = DataLoader(
            tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
            )
eval_dataloader = DataLoader(
            tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
            )





from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)






from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
            "linear",
                optimizer=optimizer,
                    num_warmup_steps=0,
                        num_training_steps=num_training_steps,
                        )
print(num_training_steps)
