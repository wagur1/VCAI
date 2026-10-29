"""Compact spatial encoder + temporal GRU predicts finite action/risk/rate."""
import torch
from torch import nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, actions, conditions=6, task_logits: bool = False):
        super().__init__(); self.actions=actions; self.conditions=conditions
        self.task_logits=bool(task_logits)
        self.encoder=nn.Sequential(nn.Conv2d(6,12,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(12,16,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        self.temporal=nn.GRU(16,24,batch_first=True)
        self.head=nn.Sequential(nn.Linear(24+conditions,32),nn.SiLU(),nn.Linear(32,actions*(7 if self.task_logits else 5)))
    def forward(self,x,maps,condition):
        if x.ndim!=5 or maps.shape!=x.shape or condition.shape!=(len(x),self.conditions):
            raise ValueError('controller condition/input shape mismatch')
        b,t,c,h,w=x.shape
        image=torch.cat((x,maps),2).reshape(b*t,6,h,w)
        if (h,w)!=(48,48):
            image=F.interpolate(image,(48,48),mode='bilinear',align_corners=False)
        z=self.encoder(image).reshape(b,t,16)
        _,hidden=self.temporal(z)
        raw=self.head(torch.cat((hidden[-1],condition),1)).reshape(b,self.actions,7 if self.task_logits else 5)
        if not self.task_logits:
            return dict(logits=raw[:,:,0],risk=raw[:,:,1:4],rate=F.softplus(raw[:,:,4])+.05)
        task=raw[:,:,:3]
        combined=(task*condition[:,None,2:5]).sum(-1)
        return dict(logits=combined,logits_task=task,risk=raw[:,:,3:6],rate=F.softplus(raw[:,:,6])+.05)


class EnsembleController(nn.Module):
    """Mean member outputs; callers must calibrate the ensemble separately."""
    def __init__(self,members,metadata=None):
        super().__init__()
        self.members=nn.ModuleList(members)
        if not self.members: raise ValueError('ensemble requires at least one controller')
        first=self.members[0]
        self.actions,self.conditions,self.task_logits=first.actions,first.conditions,first.task_logits
        if any((m.actions,m.conditions,m.task_logits)!=(self.actions,self.conditions,self.task_logits) for m in self.members):
            raise ValueError('ensemble member architecture mismatch')
        self.metadata={} if metadata is None else metadata

    def forward(self,x,maps,condition):
        outputs=[m(x,maps,condition) for m in self.members]
        if any(value.keys()!=outputs[0].keys() for value in outputs[1:]):
            raise ValueError('ensemble member output mismatch')
        return {k:torch.stack([value[k] for value in outputs]).mean(0) for k in outputs[0]}
