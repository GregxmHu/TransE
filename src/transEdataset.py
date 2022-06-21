from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import logging
import json
import random
class TransEDataset(Dataset):
    def __init__(self,datapath):
        self.datapath=datapath
        self.training_data=[]
        self.entity=[]
        self.relation=[]
        with open(self.datapath,"r") as f:
            for item in f:
                head,relation,tail=item.strip('\n').split(',')
                head,relation,tail=int(head),int(relation),int(tail)
                #head="实体："+head
                #tail="实体："+tail
                #relation="关系："+relation
                self.training_data.append(
                    {'pos':(head,relation,tail),'neg':[]}
                    )
                if head not in self.entity:
                    self.entity.append(head)
                if tail not in self.entity:
                    self.entity.append(tail)
                if relation not in self.relation:
                    self.relation.append(relation)

        self.build_training_data()
        #self.deredundancy()
        

    def __getitem__(self, idx):
        pos=self.used_training_data[idx][0]
        #neg=self.training_data[idx]['neg'].pop(0)
        #self.training_data[idx]['neg'].append(neg)
        neg=self.used_training_data[idx][1]
        #pos_h=self.tokenizer(pos[0],padding="max_length", truncation=True, max_length=10)
        #r=self.tokenizer(pos[1],padding="max_length", truncation=True, max_length=10)
        #pos_t=self.tokenizer(pos[2],padding="max_length", truncation=True, max_length=10)
        #neg_h=self.tokenizer(neg[0],padding="max_length", truncation=True, max_length=10)
        #neg_t=self.tokenizer(neg[2],padding="max_length", truncation=True, max_length=10)
        
        return (pos[0],pos[1],pos[2],neg[0],neg[2])
        #return (pos_h,r,pos_t,neg_h,neg_t)


    def __len__(self):
        return len(self.used_training_data)

    def collate(self,batch):
        pos_h=torch.tensor([item[0] for item in batch])
        r=torch.tensor([item[1] for item in batch])
        pos_t=torch.tensor([item[2] for item in batch])
        neg_h=torch.tensor([item[3] for item in batch])
        neg_t=torch.tensor([item[4] for item in batch])

        return (pos_h,r,pos_t,neg_h,neg_t)


    def deredundancy(self):
        pos=[item['pos'] for item in self.training_data]
        for index in range(len(self.training_data)):
            neg_index=0
            while True:
                if neg_index>=len(self.training_data[index]['neg']):
                    break
                if self.training_data[index]['neg'][neg_index] in pos:
                    self.training_data[index]['neg'].remove(self.training_data[index]['neg'][neg_index])
                else:
                    neg_index+=1

    def build_training_data(self):
        for index in range(len(self.training_data)):
            h,r,t=self.training_data[index]['pos']
            self.training_data[index]['neg'].extend([
                (entity,r,t) for entity in self.entity if (entity !=t and entity!=h)
            ])
            self.training_data[index]['neg'].extend([
                (h,r,entity) for entity in self.entity if (entity !=h and entity !=t)
            ])

        self.deredundancy()
        self.used_training_data=[]
        for item in self.training_data:
            pos=item['pos']
            for neg in item['neg']:
                self.used_training_data.append((pos,neg))
        random.shuffle(self.used_training_data)