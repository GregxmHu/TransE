from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import logging
import json
import random
class DevDataset(Dataset):
    def __init__(self,datapath,entity_num):
        self.datapath=datapath
        self.entity=list(range(entity_num))
        #self.relation=[]
        self.triplets=[]
        with open(self.datapath,"r") as f:
            for item in f:
                head,relation,tail=item.strip('\n').split(',')

                head,relation,tail=int(head),int(relation),int(tail)
                #head="实体："+head
                #tail="实体："+tail
                #relation="关系："+relation

                #if head not in self.entity:
                    #h_id=len(self.entity)
                    #self.entity.append(self.tokenizer(head,padding="max_length", truncation=True, max_length=10))
                    #self.entity.append(head)
                #if tail not in self.entity:
                    #t_id=len(self.entity)
                    #self.entity.append(tail)
                #if relation not in self.relation:
                    #r_id=len(self.relation)
                    #self.relation.append(relation)
                #h_id=self.entity.index(head)
                #t_id=self.entity.index(tail)
                #r_id=self.relation.index(relation)
                self.triplets.append(
                    (head,relation,tail)
                    #(h_id,r_id,t_id)
                )
                
        self.total_data=[]
        for item in self.triplets:
            for entity in range(len(self.entity)):
                if (item[0],item[1],entity) not in self.total_data:
                    if item[0] !=entity:
                        self.total_data.append((item[0],item[1],entity))
                    #print(self.total_data[-1])
                if (entity,item[1],item[2]) not in self.total_data:
                    if item[2] !=entity:
                        self.total_data.append((entity,item[1],item[2]))
                    #print(self.total_data[-1])
    def collate(self,batches):
        #batch=[item[1] for item in batches]
        #id=[item[0] for item in batches]
        #h={
        #    'input_ids':torch.tensor([item[0]['input_ids'] for item in batch]),
        #    'attention_mask':torch.tensor([item[0]['attention_mask'] for item in batch])
        #}
        #r={
        #    'input_ids':torch.tensor([item[1]['input_ids'] for item in batch]),
        #    'attention_mask':torch.tensor([item[1]['attention_mask'] for item in batch])
        #}
        #t={
        #    'input_ids':torch.tensor([item[2]['input_ids'] for item in batch]),
        #    'attention_mask':torch.tensor([item[2]['attention_mask'] for item in batch])
        #}

        #return (id,h,r,t)
        h=torch.tensor([item[0] for item in batches])
        r=torch.tensor([item[1] for item in batches])
        t=torch.tensor([item[2] for item in batches])     

        return (batches,h,r,t)
        

    def __getitem__(self, idx):

        #return self.total_data[idx],[self.tokenizer(item,padding="max_length", truncation=True, max_length=10) for item in self.total_data[idx]]
        return self.total_data[idx]
        


    def __len__(self):
        return len(self.total_data)
    