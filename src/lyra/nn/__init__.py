"""Neural-network building blocks on top of ``lyra.autograd``."""

from lyra.nn.attention import CausalSelfAttention
from lyra.nn.layers import Dropout, Embedding, LayerNorm, Linear
from lyra.nn.module import Module, ModuleList, Parameter
from lyra.nn.transformer import Block, LyraLM, ModelConfig

__all__ = [
    "Block",
    "CausalSelfAttention",
    "Dropout",
    "Embedding",
    "LayerNorm",
    "Linear",
    "LyraLM",
    "ModelConfig",
    "Module",
    "ModuleList",
    "Parameter",
]
