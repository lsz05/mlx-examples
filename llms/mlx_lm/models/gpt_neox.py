import math
from dataclasses import dataclass
from typing import Tuple, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    max_position_embeddings: int = 2048
    vocab_size: int = 51200
    hidden_size: int = 3072
    num_attention_heads = 32
    num_hidden_layers: int = 30
    num_key_value_heads: Optional[int] = None
    rotary_pct: float = 1.0  # partial_rotary_factor
    intermediate_size: int = 12288
    layer_norm_eps: float = 1e-5
    rotary_emb_base: float = 10000.0  # rope_theta
    hidden_dropout: float = 0.0

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
    

class LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)
    


class GPTNeoXAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.repeats = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rotary_emb_base
        self.partial_rotary_factor = config.rotary_pct

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
    
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.rope = nn.RoPE(
            int(self.partial_rotary_factor * self.head_dim),
            traditional=False,
            base=self.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Extract some shapes
        B, L, _ = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        def repeat(a: mx.array) -> mx.array:
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.num_heads, L, -1])


        if self.repeats > 1:
            keys, values = map(repeat, (keys, values))

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        
        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)

        # Finally preform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        
        scores = mx.softmax(scores, axis=-1).astype(values.dtype)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.dense(values_hat), (keys, values)


class GPTNeoXMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU(approx="precise")
    
    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class GPTNeoXLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.input_layernorm = LayerNorm(
            dims=config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.attention = GPTNeoXAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(
            dims=config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.mlp = GPTNeoXMLP(config)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        h = self.input_layernorm(x)
        attn_layer_outputs = self.attention(h, mask, cache)
        attn_output, cache = attn_layer_outputs[0], attn_layer_outputs[1]
        attn_output = self.post_attention_dropout(attn_output)
        
        attn_output = attn_output + x
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
        mlp_output = self.post_mlp_dropout(mlp_output)
        return mlp_output + attn_output, cache


class GPTNeoXModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [GPTNeoXLayer(config) for i in range(config.num_hidden_layers)]
        self.final_layernorm = LayerNorm(dims=config.hidden_size, eps=config.layer_norm_eps)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array]]:
        x = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)

        for i, layers in enumerate(self.layers):
            x, cache[i] = layers(x, mask, cache[i])
        
        return self.final_layernorm(x), cache


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array]]:
        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)
        
        y, cache = self.model(x, mask, cache)
        return self.embed_out(y), cache
