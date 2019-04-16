
'''
Transformer encoder for evidence-inference.

Note that this implementation largely cribbed and modified from the excellent 
"attention is all you need" implementation by Alexander Rush:

http://nlp.seas.harvard.edu/2018/04/03/attention.html
'''
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from evidence_inference.models.utils import PaddedSequence

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, input_layer, d_model, layer, N):
        super(Encoder, self).__init__()
        # embedding layer
        self.input_layer = input_layer
        # project to the model hidden dims
        self.linear_layer = nn.Linear(self.input_layer.weight.shape[1], d_model)
        # repeated application of self-attention (transformer)
        self.layers = clones(layer, N)

        self.norm = LayerNorm(layer.size)
        
    def forward(self, word_inputs, mask=None):
        "Pass the input (and mask) through each layer in turn."
        if isinstance(word_inputs, PaddedSequence):
            embedded = self.input_layer(word_inputs.data)
            x = self.linear_layer(embedded)
        
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        log_div_term = torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        div_term = torch.exp(log_div_term.float())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

import copy

class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, embeddings: nn.Embedding=None, embedding_dims=200, 
                 use_attention=False, condition_attention=False,
                 N=3, d_model=128, d_ff=256, h=8, dropout=0.1):

        super(TransformerEncoder, self).__init__()

        # this is poorly named since, of course, the transformer *always*
        # uses self-attention; this refers to token-level attention over
        # the article, which is distinct.
        self.use_attention = False 

        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        if embeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dims)
        else:
            self.embedding = embeddings


        #import pdb; pdb.set_trace()
        #input_layer = init_word_embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=inference_vectorizer.str_to_idx[inference_vectorizer.PAD], _weight=torch.FloatTensor((num_embeddings, embedding_dim)))
        
        self.condition_attention = condition_attention
        layer_to_repeat = EncoderLayer(d_model, c(attn), c(ff), dropout)
        self.model = Encoder(self.embedding, d_model, layer_to_repeat, N)

        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, word_inputs : PaddedSequence, mask=None, query_v=None):
        if self.use_attention:
            raise Error("Attention not ready for transformer yet")
        else:
            # the model will return <batch x article len x d_model> tensor.
            a_v = self.model(word_inputs, mask=mask)
            # when we are not imposing attention, we simply take the `first' 
            # transformed token representation
            a_v = a_v[:,0,:]
        return a_v


'''
def make_transformer_encoder(vocab_size, embeddings: nn.Embedding=None, N=6, d_model=32, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    if embeddings is None:
        self.embedding = nn.Embedding(vocab_size, d_model)
    else:
        self.embedding = embeddings

    #input_layer = init_word_embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=inference_vectorizer.str_to_idx[inference_vectorizer.PAD], _weight=torch.FloatTensor((num_embeddings, embedding_dim)))
    
    layer_to_repeat = EncoderLayer(d_model, c(attn), c(ff), dropout)
    model = Encoder(self.embedding, layer_to_repeat, N)

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
'''

