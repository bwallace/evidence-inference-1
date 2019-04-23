'''
A wrapper on top of the fastNLP (https://github.com/fastnlp/fastNLP/) StarTransformer
Encoder. Implementation is based on:

https://github.com/fastnlp/fastNLP/blob/master/fastNLP/modules/encoder/star_transformer.py

Modified to accomodate evidence inference API and attention.
'''

import torch
import torch.nn as nn


from evidence_inference.models.utils import PaddedSequence
from evidence_inference.models.attention_distributions import TokenAttention, evaluate_model_attention_distribution

import torch
from torch import nn
from torch.nn import functional as F
import numpy as NP


class StarTransformer(nn.Module):
    """Star-Transformer Encoder part。
    paper: https://arxiv.org/abs/1902.09113
    :param hidden_size: int, 输入维度的大小。同时也是输出维度的大小。
    :param num_layers: int, star-transformer的层数
    :param num_head: int，head的数量。
    :param head_dim: int, 每个head的维度大小。
    :param dropout: float dropout 概率
    :param max_len: int or None, 如果为int，输入序列的最大长度，
                    模型会为属于序列加上position embedding。
                    若为None，忽略加上position embedding的步骤
    """
    def __init__(self, hidden_size, num_layers, num_head, head_dim, dropout=0.1, max_len=None):
        super(StarTransformer, self).__init__()
        self.iters = num_layers

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.iters)])
        self.ring_att = nn.ModuleList(
            [MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=dropout)
                for _ in range(self.iters)])
        self.star_att = nn.ModuleList(
            [MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=dropout)
                for _ in range(self.iters)])

        if max_len is not None:
            self.pos_emb = self.pos_emb = nn.Embedding(max_len, hidden_size)
        else:
            self.pos_emb = None

    def forward(self, data, mask):
        """
        :param FloatTensor data: [batch, length, hidden] the input sequence
        :param ByteTensor mask: [batch, length] the padding mask for input, in which padding pos is 0
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        """
        def norm_func(f, x):
            # B, H, L, 1
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        B, L, H = data.size()
        mask = (mask == 0) # flip the mask for masked_fill_
        smask = torch.cat([torch.zeros(B, 1, ).byte().to(mask), mask], 1)

        embs = data.permute(0, 2, 1)[:,:,:,None] # B H L 1
        if self.pos_emb:
            P = self.pos_emb(torch.arange(L, dtype=torch.long, device=embs.device)\
                    .view(1, L)).permute(0, 2, 1).contiguous()[:, :, :, None]  # 1 H L 1
            embs = embs + P

        nodes = embs
        relay = embs.mean(2, keepdim=True)
        ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        r_embs = embs.view(B, H, 1, L)
        for i in range(self.iters):
            ax = torch.cat([r_embs, relay.expand(B, H, 1, L)], 2)
            nodes = nodes + F.leaky_relu(self.ring_att[i](norm_func(self.norm[i], nodes), ax=ax))
            relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 2), smask))

            nodes = nodes.masked_fill_(ex_mask, 0)

        nodes = nodes.view(B, H, L).permute(0, 2, 1)

        return nodes, relay.view(B, H)


class MSA1(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        super(MSA1, self).__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, ax=None):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,H,L,1)

        if ax is not None:
            aL = ax.shape[2]
            ak = self.WK(ax).view(B, nhead, head_dim, aL, L)
            av = self.WV(ax).view(B, nhead, head_dim, aL, L)
        q = q.view(B, nhead, head_dim, 1, L)
        k = F.unfold(k.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0))\
                .view(B, nhead, head_dim, unfold_size, L)
        v = F.unfold(v.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0))\
                .view(B, nhead, head_dim, unfold_size, L)
        if ax is not None:
            k = torch.cat([k, ak], 3)
            v = torch.cat([v, av], 3)

        alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / NP.sqrt(head_dim), 3))  # B N L 1 U
        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)

        ret = self.WO(att)

        return ret


class MSA2(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(MSA2, self).__init__()
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h
        pre_a = torch.matmul(q, k) / NP.sqrt(head_dim)
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float('inf'))
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        return self.WO(att)


class StarTransformerEncoder(nn.Module):

    def __init__(self, vocab_size, embeddings: nn.Embedding=None, 
                 use_attention=False, condition_attention=False, tokenwise_attention=False, query_dims=None,
                 N=3, d_model=128, h=4, dropout=0.1, concat_relay=False):

        super(StarTransformerEncoder, self).__init__()

        self.d_model = d_model # hidden dims for transformer

        # the use_attention flag determines whether we impose attention over
        # *tokens*, which is independent of the self-attention mechanism
        # used by the transformer
        self.use_attention = use_attention
        self.condition_attention = condition_attention
        self.query_dims = query_dims

        attention_input_dims = self.d_model 

        # if this is true, then we concatenate the relay node to all token embeddings
        self.concat_relay = concat_relay 
        if self.concat_relay:
            attention_input_dims = attention_input_dims + self.d_model

        if self.use_attention:
            self.attention_mechanism = TokenAttention(attention_input_dims, self.query_dims, condition_attention, tokenwise_attention)
        
        if embeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dims)
        else:
            self.embedding = embeddings

        # we need to map word embedding inputs to transformer hidden dims 
        self.projection_layer = nn.Linear(self.embedding.weight.shape[1], d_model)

        # 'hidden_size', 'num_layers', 'num_head', 'head_dim'
        # @TODO I do not understand what head_dim is... 
        self.st = StarTransformer(d_model, N, h, d_model)


    def _concat_relay_to_tokens_in_batches(self, article_token_batches, relay_batches, batch_sizes):
        '''
        Takes <batch x doc_len x embedding> tensor (article_token_batches) and builds and returns
        a version <batch x doc_len x [embedding + relay_embedding]> which concatenates repeated
        copies of the relay embedding associated with each batch.
        '''
        
        # create an empty <batch x (token emedding + relay_embedding)> 
        article_tokens_with_relays = torch.zeros(article_token_batches.data.shape[0], 
                                                 article_token_batches.data.shape[1],
                                                 article_token_batches.data.shape[2] + relay_batches.shape[1])

        for b in range(article_token_batches.data.shape[0]):
            batch_relay = relay_batches[b].repeat(article_tokens_with_relays.shape[1], 1)
            article_tokens_with_relays[b] = torch.cat((article_token_batches.data[b], batch_relay), 1)

        return PaddedSequence(article_tokens_with_relays.to("cuda"), batch_sizes, batch_first=True)

    def forward(self, word_inputs : PaddedSequence, mask=None, query_v_for_attention=None, normalize_attention_distribution=True):
           
        embedded = self.embedding(word_inputs.data)
        projected = self.projection_layer(embedded)
        mask = word_inputs.mask().to("cuda")

        # now to the star transformer.
        # the model will return a tuple comprising <batch, words, dims> and a second
        # tensor (the rely nodes) of <batch, dims> -- we take the latter
        # in the case where no attention is to be used
        token_vectors, a_v = self.st(projected, mask=mask) 
        

        if self.use_attention:
            token_vectors = PaddedSequence(token_vectors, word_inputs.batch_sizes, batch_first=True)
            a = None
            if self.concat_relay:
                ###
                # need to concatenate a_v <batch x model_d> for all articles
                ###
                token_vectors_with_relay = self._concat_relay_to_tokens_in_batches(token_vectors, a_v, word_inputs.batch_sizes)
                
                a = self.attention_mechanism(token_vectors_with_relay, query_v_for_attention, normalize=normalize_attention_distribution)
            else:
                a = self.attention_mechanism(token_vectors, query_v_for_attention, normalize=normalize_attention_distribution)
          
            # note this is an element-wise multiplication, so each of the hidden states is weighted by the attention vector
            weighted_hidden = torch.sum(a * token_vectors.data, dim=1)
        
            return token_vectors, weighted_hidden, a

        return a_v
