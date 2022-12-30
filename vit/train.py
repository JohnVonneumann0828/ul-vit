import wandb
import torch
import vit
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import argparse
import numpy as np
import random 
import os
import pickle
from transformers import BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=100)
parser.add_argument("--train_batch_size", type=int, default=1)
parser.add_argument("--dev_batch_size", type=int, default=1)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased", "xlnet-base-cased"],
    default="bert-base-uncased",
)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
#parser.add_argument("--seed", type=seed, default="random")


args, unknown = parser.parse_known_args()
model=vit.VisionTransformer(hidden_dims=[768])
DEVICE="cuda:0"
amt=0
curr_index=0
pathway="e:\\project\\Prepared_dataset\\Dataset\\"
c=open(pathway+"intervals.txt","r")
c=c.read()
c=c.split("\n")
#print(c)
for i in range(len(c)):
    c[i]=int(c[i])
#print(c)
tot=sum(c)
d=open(pathway+"labels.pkl","rb")
labels=pickle.load(d)
d.close()
a=open(pathway+"video_raw_12_13.pkl","rb") #Opening the pkl file with 62 gigs data
tr_loss = 0
nb_tr_examples, nb_tr_steps = 0, 0
def prep_for_training(num_train_optimization_steps: int):

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    return optimizer, scheduler
def train(optimizer,scheduler):
  amt=0
  curr_index=0
  while True:
      try:
          data=pickle.load(a)
          entry=[]
          amt+=1
          if(amt>=c[curr_index]):
              amt=0
              curr_index+=1
              if(curr_index>=len(c)):
                  break
          #print(data)
          data1=torch.tensor(data).float()
          #data1.unsqueeze(0)
          data1=torch.unsqueeze(data1,0)
          print(data1.dim())
          outputs=model(data1)
          label_ids=[labels[curr_index]]
          logits = outputs[0]

          loss_fct = MSELoss()
          loss = loss_fct(logits.view(-1), label_ids.view(-1))

          loss.backward()
          if args.gradient_accumulation_step > 1:
              loss = loss / args.gradient_accumulation_step

          tr_loss += loss.item()
          nb_tr_steps += 1

          if (step + 1) % args.gradient_accumulation_step == 0:
              optimizer.step()
              scheduler.step()
              optimizer.zero_grad()
        
      except EOFError:
        return loss/nb_tr_steps
        break
optimizer,scheduler=prep_for_training(tot)
a=train(optimizer,scheduler)
