"""Raw-space transfer maps for Gate 2C-VM896.

The module fits recipient concept frames directly from the original 896-wide
Qwen states. It keeps centering, whitening, width, global/nonlinear transfer,
and MACSL local residual ownership experimentally separable.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, math
from typing import Any, Iterable, Sequence
import numpy as np
import torch
from torch import nn
from .vivere_macsl import KMeansRouter


def _rows(values: Sequence[int] | np.ndarray) -> np.ndarray:
    result=np.asarray(values,dtype=np.int64)
    if result.ndim!=1 or len(result)==0: raise ValueError('rows must be nonempty 1D')
    return result

def _safe_scale(values: np.ndarray) -> np.ndarray:
    scale=np.asarray(values,dtype=np.float32)
    return np.where(np.abs(scale)<1e-6,1.0,scale).astype(np.float32)

@dataclass(frozen=True)
class DefinitionSplit:
    fit: np.ndarray; tune: np.ndarray; test: np.ndarray
    @classmethod
    def from_training_validation(cls,training_rows:Sequence[int],validation_rows:Sequence[int],*,tune_rows:int=800,salt:str='gate2c-vm896-definition-v1')->'DefinitionSplit':
        training=list(map(int,training_rows)); training.sort(key=lambda r:hashlib.sha256(f'{salt}:{r}'.encode()).digest())
        if not 1<=tune_rows<len(training): raise ValueError('invalid tune_rows')
        return cls(np.asarray(sorted(training[tune_rows:]),dtype=np.int64),np.asarray(sorted(training[:tune_rows]),dtype=np.int64),np.asarray(sorted(map(int,validation_rows)),dtype=np.int64))

@dataclass(frozen=True)
class ReferenceFrame:
    mean:np.ndarray; basis:np.ndarray; scale:np.ndarray; centered:bool; whitened:bool; fit_rows:int; singular_values:np.ndarray; explained_variance_fraction:float
    @classmethod
    def fit(cls,reference:np.ndarray,rows:Sequence[int]|np.ndarray,*,width:int,centered:bool=True,whitened:bool=False)->'ReferenceFrame':
        matrix=np.asarray(reference,dtype=np.float32); selected=_rows(rows)
        if matrix.ndim!=2 or not 1<=width<=min(matrix.shape): raise ValueError('invalid frame width')
        mean=matrix[selected].mean(0).astype(np.float32) if centered else np.zeros(matrix.shape[1],dtype=np.float32)
        training=(matrix[selected]-mean).astype(np.float64)
        _,singular,right=np.linalg.svd(training,full_matrices=False); right=right[:width].astype(np.float32)
        for i in range(width):
            pivot=int(np.argmax(np.abs(right[i])))
            if right[i,pivot]<0: right[i]*=-1
        basis=right.T.copy(); scores=training@basis.astype(np.float64)
        scale=_safe_scale(scores.std(0)) if whitened else np.ones(width,dtype=np.float32)
        centered_training=matrix[selected]-matrix[selected].mean(0); total=float(np.square(centered_training).sum())
        explained=float(np.square(singular[:width]).sum()/max(total,1e-12))
        return cls(mean,basis,scale,bool(centered),bool(whitened),len(selected),singular[:width].astype(np.float32),explained)
    @property
    def width(self)->int:return int(self.basis.shape[1])
    def transform(self,values:np.ndarray)->np.ndarray:return (((np.asarray(values,dtype=np.float32)-self.mean)@self.basis)/self.scale).astype(np.float32)
    def to_record(self)->dict[str,Any]:return {'width':self.width,'centered':self.centered,'whitened':self.whitened,'fit_rows':self.fit_rows,'explained_variance_fraction':self.explained_variance_fraction,'singular_values':self.singular_values.astype(float).tolist()}

@dataclass(frozen=True)
class SourceStandardizer:
    mean:np.ndarray; scale:np.ndarray
    @classmethod
    def fit(cls,values:np.ndarray,rows:Sequence[int]|np.ndarray)->'SourceStandardizer':
        selected=np.asarray(values,dtype=np.float32)[_rows(rows)]
        return cls(selected.mean(0).astype(np.float32),_safe_scale(selected.std(0)))
    def transform(self,values:np.ndarray)->np.ndarray:return ((np.asarray(values,dtype=np.float32)-self.mean)/self.scale).astype(np.float32)

@dataclass(frozen=True)
class RidgeSolution:
    standardizer:SourceStandardizer; target_mean:np.ndarray; weight:np.ndarray; ridge:float; validation_mse:float
    def predict(self,values:np.ndarray)->np.ndarray:return (self.standardizer.transform(values)@self.weight+self.target_mean).astype(np.float32)

class SpectralRidgeFactory:
    def __init__(self,source:np.ndarray,rows:Sequence[int]|np.ndarray)->None:
        self.source=np.asarray(source,dtype=np.float32); self.rows=_rows(rows); self.standardizer=SourceStandardizer.fit(self.source,self.rows); self.normalized=self.standardizer.transform(self.source)
        self.u,self.singular,self.right=np.linalg.svd(self.normalized[self.rows].astype(np.float64),full_matrices=False)
    def latent(self,values:np.ndarray|None=None,*,width:int=32)->np.ndarray:
        matrix=self.normalized if values is None else self.standardizer.transform(values); width=min(int(width),self.right.shape[0]); return (matrix@self.right[:width].T.astype(np.float32)).astype(np.float32)
    def fit(self,target:np.ndarray,tune_rows:Sequence[int]|np.ndarray,*,candidates:Iterable[float]=(0.1,1.0,10.0,30.0,100.0,300.0))->tuple[RidgeSolution,list[dict[str,float]]]:
        target=np.asarray(target,dtype=np.float32); tune=_rows(tune_rows); mean=target[self.rows].mean(0).astype(np.float32); y=(target[self.rows]-mean).astype(np.float64); uty=self.u.T@y; trials=[]; best=None
        for ridge in candidates:
            factors=self.singular/(np.square(self.singular)+float(ridge)); weight=(self.right.T@(factors[:,None]*uty)).astype(np.float32); pred=self.normalized[tune]@weight+mean; score=float(np.square(pred-target[tune]).mean()); trials.append({'ridge':float(ridge),'validation_mse':score}); candidate=RidgeSolution(self.standardizer,mean,weight,float(ridge),score)
            if best is None or score<best.validation_mse:best=candidate
        assert best is not None; return best,trials

class _TransferMLP(nn.Module):
    def __init__(self,source_width:int,target_width:int,hidden_width:int)->None:
        super().__init__(); self.network=nn.Sequential(nn.Linear(source_width,hidden_width),nn.GELU(),nn.Linear(hidden_width,target_width))
    def forward(self,values:torch.Tensor)->torch.Tensor:return self.network(values)
@dataclass(frozen=True)
class NonlinearSolution:
    standardizer:SourceStandardizer; state:dict[str,np.ndarray]; source_width:int; target_width:int; hidden_width:int; best_validation_mse:float; epochs:int
    def predict(self,values:np.ndarray)->np.ndarray:
        model=_TransferMLP(self.source_width,self.target_width,self.hidden_width); model.load_state_dict({n:torch.from_numpy(v.copy()) for n,v in self.state.items()}); model.eval()
        with torch.no_grad():return model(torch.from_numpy(self.standardizer.transform(values))).cpu().numpy().astype(np.float32)
def fit_nonlinear(source:np.ndarray,target:np.ndarray,split:DefinitionSplit,*,seed:int,hidden_width:int=128,maximum_epochs:int=300,patience:int=35)->NonlinearSolution:
    torch.manual_seed(int(seed)); standardizer=SourceStandardizer.fit(source,split.fit); x=torch.from_numpy(standardizer.transform(source)); y=torch.from_numpy(np.asarray(target,dtype=np.float32)); model=_TransferMLP(x.shape[1],y.shape[1],hidden_width); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); best_score=math.inf; best_state=None; stale=0; epoch=0
    for epoch in range(1,int(maximum_epochs)+1):
        model.train(); loss=torch.nn.functional.mse_loss(model(x[split.fit]),y[split.fit]); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); model.eval()
        with torch.no_grad():score=float(torch.nn.functional.mse_loss(model(x[split.tune]),y[split.tune]))
        if score<best_score-1e-7:best_score=score;best_state={n:v.detach().cpu().numpy().copy() for n,v in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=patience:break
    assert best_state is not None; return NonlinearSolution(standardizer,best_state,x.shape[1],y.shape[1],int(hidden_width),best_score,epoch)

@dataclass(frozen=True)
class LocalResidualCard:
    residual_mean:np.ndarray; residual_basis:np.ndarray; latent_mean:np.ndarray; coefficient_mean:np.ndarray; coefficient_weight:np.ndarray; shrinkage:float
    def predict(self,latent:np.ndarray)->np.ndarray:
        coefficient=(latent-self.latent_mean)@self.coefficient_weight+self.coefficient_mean; return (float(self.shrinkage)*(self.residual_mean+coefficient@self.residual_basis)).astype(np.float32)
@dataclass(frozen=True)
class MacslSolution:
    global_map:RidgeSolution; source_factory:SpectralRidgeFactory; router:KMeansRouter; cards:tuple[LocalResidualCard,...]; top_k:int; source_latent_width:int; validation_mse:float
    def predict(self,source:np.ndarray,router_source:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        indices,weights=self.router.route(router_source,top_k=self.top_k); latent=self.source_factory.latent(source,width=self.source_latent_width); local=np.zeros((len(source),self.top_k,self.global_map.weight.shape[1]),dtype=np.float32)
        for slot in range(self.top_k):
            for card_index,card in enumerate(self.cards):
                selected=np.flatnonzero(indices[:,slot]==card_index)
                if len(selected):local[selected,slot]=card.predict(latent[selected])
        return self.global_map.predict(source)+(local*weights[...,None]).sum(1),indices,weights
def fit_macsl(source:np.ndarray,target:np.ndarray,router_source:np.ndarray,split:DefinitionSplit,*,source_factory:SpectralRidgeFactory,global_map:RidgeSolution,card_count:int,top_k:int,residual_rank:int=8,source_latent_width:int=32,local_ridge:float=30.0,shrinkage_prior:float=40.0,seed:int=20260804,router:KMeansRouter|None=None)->MacslSolution:
    router=router or KMeansRouter.fit(router_source,split.fit,card_count=int(card_count),seed=int(seed)); assignment=router.distances(router_source).argmin(1); global_prediction=global_map.predict(source); residual_all=np.asarray(target,dtype=np.float32)-global_prediction; latent_all=source_factory.latent(source,width=int(source_latent_width)); cards=[]
    for card_index in range(int(card_count)):
        rows=split.fit[assignment[split.fit]==card_index]
        if len(rows)<5:cards.append(LocalResidualCard(np.zeros(target.shape[1],np.float32),np.zeros((1,target.shape[1]),np.float32),np.zeros(source_latent_width,np.float32),np.zeros(1,np.float32),np.zeros((source_latent_width,1),np.float32),0.0));continue
        residual=residual_all[rows]; residual_mean=residual.mean(0).astype(np.float32); centered=(residual-residual_mean).astype(np.float64); _,_,right=np.linalg.svd(centered,full_matrices=False); rank=max(1,min(int(residual_rank),right.shape[0],len(rows)-1)); basis=right[:rank].astype(np.float32); coefficients=(centered@basis.T).astype(np.float64); latent=latent_all[rows].astype(np.float64); latent_mean=latent.mean(0).astype(np.float32); coefficient_mean=coefficients.mean(0).astype(np.float32); xc=latent-latent_mean; yc=coefficients-coefficient_mean; weight=np.linalg.solve(xc.T@xc+float(local_ridge)*np.eye(xc.shape[1]),xc.T@yc).astype(np.float32); shrinkage=float(len(rows)/(len(rows)+float(shrinkage_prior))); cards.append(LocalResidualCard(residual_mean,basis,latent_mean,coefficient_mean,weight,shrinkage))
    provisional=MacslSolution(global_map,source_factory,router,tuple(cards),int(top_k),int(source_latent_width),math.inf); prediction,_,_=provisional.predict(source,router_source); score=float(np.square(prediction[split.tune]-target[split.tune]).mean()); return MacslSolution(global_map,source_factory,router,tuple(cards),int(top_k),int(source_latent_width),score)
def card_utilization(indices:np.ndarray,rows:Sequence[int],card_count:int)->dict[str,Any]:
    selected=np.asarray(indices,dtype=np.int64)[_rows(rows)];counts=np.bincount(selected.reshape(-1),minlength=int(card_count));probability=counts/max(float(counts.sum()),1.0);active=probability>0;entropy=float(-(probability[active]*np.log(probability[active])).sum()/math.log(card_count)) if card_count>1 and active.any() else 0.0;return {'counts':counts.astype(int).tolist(),'active_cards':int((counts>0).sum()),'normalized_entropy':entropy}
