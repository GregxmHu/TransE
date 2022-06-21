from torch import nn
import torch
class TransE(nn.Module):
    def __init__(self,entity_num,relation_num,dim_num):
        super(TransE,self).__init__()
        self.entity_embedding=nn.Embedding(entity_num,dim_num)
        self.relation_embedding=nn.Embedding(relation_num,dim_num)

        #entity_data=torch.rand(entity_num,dim_num)*(12/torch.sqrt(torch.tensor(dim_num)))-6/torch.sqrt(torch.tensor(dim_num))
        #relation_data=torch.rand(relation_num,dim_num)*(12/torch.sqrt(torch.tensor(dim_num)))-6/torch.sqrt(torch.tensor(dim_num))
        #relation_data=relation_data/torch.norm(relation_data,p=2,dim=-1).reshape(-1,1)
        #entity_data=entity_data/torch.norm(entity_data,p=2,dim=-1).reshape(-1,1)
        #self.entity_embedding.weight.data=entity_data.clone().detach()
        #self.relation_embedding.weight.data=relation_data.clone().detach()

        self.entity_embedding.weight.data.uniform_(-6/torch.sqrt(torch.tensor(dim_num)),6/torch.sqrt(torch.tensor(dim_num)))
        self.relation_embedding.weight.data.uniform_(-6/torch.sqrt(torch.tensor(dim_num)),6/torch.sqrt(torch.tensor(dim_num)))
        self.relation_embedding.weight.data=self.relation_embedding.weight.data/torch.norm(self.relation_embedding.weight.data,p=2,dim=-1).reshape(-1,1)
        self.entity_embedding.weight.data=self.entity_embedding.weight.data/torch.norm(self.entity_embedding.weight.data,p=2,dim=-1).reshape(-1,1)
    def forward(self,input_ids,entity:bool=False):
        if entity:
            output=self.entity_embedding(input_ids)
            #stand_output=output/torch.norm(output,p=2,dim=-1).reshape(-1,1)
            return output
            #return output
        else:
            output=self.relation_embedding(input_ids)
            #stand_output=output/torch.norm(output,p=2,dim=-1).reshape(-1,1)
            return output

    def normlize(self):
        self.entity_embedding.weight.data=self.entity_embedding.weight.data/torch.norm(self.entity_embedding.weight.data,p=2,dim=-1).reshape(-1,1)

