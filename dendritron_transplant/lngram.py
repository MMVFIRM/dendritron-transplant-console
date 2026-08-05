"""Compact hard latent n-gram memory with an explicit surrogate route gradient."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .config import LNGramConfig


def _pack_bits(bits: Tensor) -> Tensor:
    powers = (1 << torch.arange(bits.shape[-1], device=bits.device)).to(torch.long)
    return (bits.to(torch.long) * powers).sum(dim=-1)


def _addresses(symbols: Tensor, order: int, alphabet: int) -> tuple[Tensor, Tensor]:
    batch,length,routes=symbols.shape
    addresses=torch.zeros_like(symbols)
    valid=torch.zeros_like(symbols,dtype=torch.bool)
    if length < order:
        return addresses,valid
    ends=length-order+1
    route_offset=(torch.arange(routes,device=symbols.device)*alphabet**order).view(1,1,routes)
    local=route_offset.expand(batch,ends,routes).clone()
    for position in range(order):
        local += symbols[:,position:position+ends] * alphabet**position
    addresses[:,order-1:]=local; valid[:,order-1:]=True
    return addresses,valid


@dataclass(frozen=True)
class LNGramStats:
    symbols: Tensor
    addresses: dict[int,Tensor]
    valid: dict[int,Tensor]


class LNGramMemory(nn.Module):
    def __init__(self, width: int, config: LNGramConfig, *, initialization_gain: float = 1.0) -> None:
        super().__init__()
        if width % config.bits_per_route:
            raise ValueError("width must be divisible by LNGram bits_per_route")
        self.width=width; self.config=config
        self.routes=width//config.bits_per_route; self.alphabet=2**config.bits_per_route
        self.address_projection=nn.Linear(width,width,bias=False)
        self.tables=nn.ParameterDict()
        self.value_projection=nn.ModuleDict()
        retrieval_width=self.routes*config.route_memory_width
        for order in config.orders:
            rows=self.routes*self.alphabet**order
            table=nn.Parameter(torch.empty(rows,config.route_memory_width))
            nn.init.normal_(table,std=width**-0.5); self.tables[str(order)]=table
            projection=nn.Linear(retrieval_width,width,bias=False)
            with torch.no_grad(): projection.weight.mul_(initialization_gain)
            self.value_projection[str(order)]=projection
        self.surrogate=nn.Linear(width,width,bias=False)
        self.output_gate=nn.Parameter(torch.tensor(1e-3))

    @property
    def stored_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def active_parameters_per_position(self) -> int:
        table_values=self.routes*self.config.route_memory_width*len(self.config.orders)
        projections=sum(module.weight.numel() for module in self.value_projection.values())
        return self.address_projection.weight.numel()+table_values+projections+self.surrogate.weight.numel()+1

    def forward(self, hidden: Tensor, *, return_stats: bool=False) -> Tensor | tuple[Tensor,LNGramStats]:
        logits=self.address_projection(F.normalize(hidden,dim=-1)).view(
            *hidden.shape[:2],self.routes,self.config.bits_per_route)
        probabilities=torch.sigmoid(logits)
        symbols=_pack_bits(probabilities>0.5)
        updates=[]; address_stats={}; valid_stats={}
        for order in self.config.orders:
            addresses,valid=_addresses(symbols,order,self.alphabet)
            values=self.tables[str(order)].index_select(0,addresses.reshape(-1)).view(
                *addresses.shape,self.config.route_memory_width)
            values=values*valid.unsqueeze(-1)
            updates.append(self.value_projection[str(order)](values.flatten(start_dim=-2)))
            address_stats[order]=addresses; valid_stats[order]=valid
        update=sum(updates)/len(updates)
        soft=probabilities.flatten(start_dim=-2)
        surrogate=self.surrogate(soft)
        update=update+self.config.surrogate_scale*(surrogate-surrogate.detach())
        output=torch.tanh(self.output_gate)*update
        if return_stats:
            return output,LNGramStats(symbols,address_stats,valid_stats)
        return output
