'''
A very thin wrapper on top of the fastNLP (https://github.com/fastnlp/fastNLP/) StarTransformer
Encoder just to comply with the evidence inference interface.
'''

import torch
import torch.nn as nn

# pip install fastNLP; https://github.com/fastnlp/fastNLP/
from fastNLP.modules.encoder import star_transformer

from evidence_inference.models.utils import PaddedSequence


class StarTransformerEncoder(nn.Module):

    def __init__(self, vocab_size, embeddings: nn.Embedding=None, embedding_dims=200, 
                 use_attention=False, condition_attention=False,
                 N=6, d_model=128, h=8, dropout=0.1):

        super(StarTransformerEncoder, self).__init__()

        # this is poorly named since, of course, the transformer *always*
        # uses self-attention; this refers to token-level attention over
        # the article, which is distinct.
        self.use_attention = False 
        self.d_model = d_model # hidden dims for transformer

        if embeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dims)
        else:
            self.embedding = embeddings

        # we need to map word embedding inputs to transformer hidden dims 
        self.projection_layer = nn.Linear(self.embedding.weight.shape[1], d_model)

        # 'hidden_size', 'num_layers', 'num_head', 'head_dim'
        # @TODO I do not understand what head_dim is... 
        self.st = star_transformer.StarTransformer(d_model, N, h, d_model)



    def forward(self, word_inputs : PaddedSequence, mask=None, query_v=None):
        if self.use_attention:
            raise Error("Attention not ready for star transformer yet")
        else:
            
            embedded = self.embedding(word_inputs.data)
            projected = self.projection_layer(embedded)
            mask = word_inputs.mask().to("cuda")


            # now to the star transformer
            # the model will return a tuple comprising <batch, words, dims> and a second
            # tensor (the rely nodes) of <batch, dims> -- we take the latter.
            _, a_v = self.st(projected, mask=mask) 
             
            
        return a_v
