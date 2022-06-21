import argparse
import gzip
import json
import logging
import os
import pickle
import random
import tarfile
from datetime import datetime
from transformers import AutoTokenizer,BertModel
from transformers import get_linear_schedule_with_warmup
from transEdataset import TransEDataset
from Devdataset import DevDataset
from encode import TransE
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch.cuda
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.optim as optim
import time
#### Just some code to print debug information to stdout
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
def dev(model,dev_dataloader):
    model.eval()
    results={}
    with torch.no_grad():
        for triplet in tqdm(dev_dataloader):
            id,h,r,t=triplet[0],triplet[1],triplet[2],triplet[3]
            #print(h,r,t)
            h_embed=model(h.to(device),True)
            r_embed=model(r.to(device),False)
            t_embed=model(t.to(device),True)
            score=torch.norm(h_embed+r_embed-t_embed,dim=-1,p=args.norm_number).tolist()
            for iden in range(len(score)):
                results[id[iden]]=score[iden]
    #print(results)

    #print(results)
    hits10,mr,mrr=get_metrics(results,dev_dataset.triplets)
    #print(mr)
    #print("hits@10:",hits10/len(mr))
    mrr=[v for v in mrr.values()]
    #print("mrr:",sum(mrr)/len(mrr))
    return sum(mrr)/len(mrr)

def get_metrics(results,triplets):
    hits10=0
    mrr={}
    mr={}
    #print(results,triplets)
    for triplet in triplets:
        eval_results={}
        for item in results:
            if (item[2]==triplet[2] and item[1]==triplet[1]): #or (item[2]==triplet[2] and item[1]==triplet[1]) :
                eval_results[item]=results[item]
        sorted_results=sorted(eval_results.items(),key=lambda x: x[1])
        #print(sorted_results)
            #sorted_results={k:results[k] for k in sorted_results}
            #print(sorted_results)
        for i in range(len(sorted_results)):
            if sorted_results[i][0]==triplet:
                if i<10:
                    hits10+=1
                mr[triplet]=i+1
                mrr[triplet]=1/(i+1)
                break
    return hits10,mr,mrr

device=torch.device("cuda:2")
parser = argparse.ArgumentParser()
parser.add_argument("--train_data_path", default=None, type=str)
parser.add_argument("--test_data_path", default=None, type=str)
parser.add_argument("--checkpoint_save_path", default=None, type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--log_dir", type=str,default=None)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--warmup_steps", default=0, type=int)
parser.add_argument("--norm_number", default=2, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--seed", default=13, type=int)
args = parser.parse_args()



random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

writer = SummaryWriter(args.log_dir)
train_dataset=TransEDataset(args.train_data_path)
dev_dataset=DevDataset(args.train_data_path,15)
test_dataset=DevDataset(args.test_data_path,15)
#model=BertModel.from_pretrained(args.pretrained_model_name_or_path)
model=TransE(entity_num=15,relation_num=23,dim_num=100).to(device)
#optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
optimizer=optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
loss_fn=nn.MarginRankingLoss(1).to(device)
train_dataloader=DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=12,collate_fn=train_dataset.collate)
dev_dataloader=DataLoader(dataset=dev_dataset,batch_size=args.batch_size,shuffle=False,num_workers=10,collate_fn=dev_dataset.collate)
test_dataloader=DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=10,collate_fn=test_dataset.collate)
scheduler=get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs*len(train_dataloader))
model.zero_grad()
model.train
best_mrr=0.0
for epoch in tqdm(range(args.epochs)):
    for i,(pos_h,r,pos_t,neg_h,neg_t) in tqdm(enumerate(train_dataloader)):
        pos_h_embeddings=model(pos_h.to(device),True)

        r_embeddings=model(pos_h.to(device),False)

        pos_t_embeddings=model(pos_t.to(device),True)

        neg_h_embeddings=model(neg_h.to(device),True)

        neg_t_embeddings=model(neg_t.to(device),True)

        true_dis=torch.norm(pos_h_embeddings+r_embeddings-pos_t_embeddings,p=args.norm_number,dim=-1)
        false_dis=torch.norm(neg_h_embeddings+r_embeddings-neg_t_embeddings,p=args.norm_number,dim=-1)
        target=torch.ones_like(false_dis)*(-1)
        loss=loss_fn(true_dis,false_dis,target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        model.normlize()
    mrr=dev(model,dev_dataloader)
    writer.add_scalar("dev_mrr", mrr, epoch + 1)
    if mrr>best_mrr:
        best_mrr=mrr
        torch.save(model.state_dict(),args.checkpoint_save_path)
        print("mrr",mrr)
    model.train()

model.load_state_dict(torch.load(args.checkpoint_save_path))
model.eval()
results={}
with torch.no_grad():
    for triplet in tqdm(test_dataloader):
        id,h,r,t=triplet[0],triplet[1],triplet[2],triplet[3]
        #print(h,r,t)
        h_embed=model(h.to(device),True)
        r_embed=model(r.to(device),False)
        t_embed=model(t.to(device),True)
        score=torch.norm(h_embed+r_embed-t_embed,dim=-1,p=args.norm_number).tolist()
        for iden in range(len(score)):
            results[id[iden]]=score[iden]
#print(results)

#print(results)
hits10,mr,mrr=get_metrics(results,test_dataset.triplets)
print(mr)
print("hits@10:",hits10/len(mr))
mrr=[v for v in mrr.values()]
print("mrr:",sum(mrr)/len(mrr))