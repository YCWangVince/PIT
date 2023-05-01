# try:
# from .layers import *
# except:
#     from galerkin_transformer.layers import *
#     from galerkin_transformer.utils_ft import *

import copy
import os
import sys
from collections import defaultdict
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MultiheadAttention, TransformerEncoderLayer
from torch.nn.init import constant_, xavier_uniform_
from torchinfo import summary

current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
sys.path.append(SRC_ROOT)
#
# ADDITIONAL_ATTR = [ 'return_latent', 'residual_type', 'norm_type', 'norm_eps', 'boundary_condition',
#                     'spacial_dim', 'spacial_fc', 'encoder_dropout', 'ffn_dropout', 'decoder_dropout']


def return_activation(activation_name):
    # return activation nn
    activation_list = ['tanh', 'relu', 'leaky_relu', 'silu']
    assert activation_name in activation_list, "Only support tanh, relu, leaky_relu, and silu"

    activation_nn = [nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.SiLU]
    if activation_name in activation_list:
        return activation_nn[activation_list.index(activation_name)]

def attention(query, key, value, attention_type='softmax'):
    '''
    Simplified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    Compute the Scaled Dot Product Attention
    '''

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    seq_len = scores.size(-1)

    assert attention_type in ['softmax', 'fourier', 'fourier_zero_sum']

    if attention_type == 'softmax':
        p_attn = F.softmax(scores, dim=-1)
    elif attention_type == 'fourier':
        p_attn = scores / seq_len
    elif attention_type == 'fourier_zero_sum':
        # scores *= math.sqrt(d_k)
        p_attn = (scores / seq_len - torch.mean(scores / seq_len, dim=-1, keepdim=True))

    out = torch.matmul(p_attn, value)

    return out, p_attn

def linear_attention(query, key, value, attention_type='galerkin'):
    '''
    Adapted from lucidrains' implementaion
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    linear_attn function
    Compute the Scaled Dot Product Attention globally
    '''

    seq_len = query.size(-2)
    scores = torch.matmul(key.transpose(-2, -1), value)
    assert attention_type == 'galerkin'
    p_attn = scores / seq_len
    out = torch.matmul(query, p_attn)
    return out, p_attn

class Attention(nn.Module):
    '''
    The attention is using a vanilla (QK^T)V or Q(K^T V)
    For an encoder layer, the tensor size is slighly different from the official pytorch implementation

    attn_types:
        - fourier: integral, local
        - fourier_zero_sum: integral for zero mean
        - galerkin: global
        - softmax: classic softmax attention

    In this implementation, output is (N, L, E).
    batch_first will be added in the next version of PyTorch: https://github.com/pytorch/pytorch/pull/55285

    Reference: code base modified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - added layer norm <-> attn norm switch
    - added diagonal init

    In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    the linear attention in each head is implemented as an Einstein sum
    attn_matrix = torch.einsum('bhnd,bhne->bhde', k, v)
    attn = torch.einsum('bhnd,bhde->bhne', q, attn_matrix)
    return attn.reshape(*q.shape)
    here in our implementation this is achieved by a slower transpose+matmul
    but can conform with the template Harvard NLP gave
    '''

    def __init__(self, n_head, d_model,
                 attention_type='fourier',
                 symmetric_init=False,
                 norm=False,
                 norm_type='layer',
                 eps=1e-5):
        super(Attention, self).__init__()
        assert d_model % n_head == 0
        self.attention_type = attention_type
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.symmetric_init = symmetric_init
        self.add_norm = norm
        self.norm_type = norm_type
        if norm:
            self._get_norm(eps=eps)

        self.attn_weight = None

    def forward(self, query, key, value):

        bsz = query.size(0)

        query, key, value = \
            [layer(x).view(bsz, self.n_head, self.d_k).transpose(0, 1)
             for layer, x in zip(self.linears, (query, key, value))]

        if self.add_norm:
            if self.attention_type == 'galerkin':
                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                value = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)
            else:
                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                query = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_Q, (query[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

        if self.attention_type == 'galerkin':
            x, self.attn_weight = linear_attention(query, key, value, attention_type=self.attention_type)
        else:
            x, self.attn_weight = attention(query, key, value, attention_type=self.attention_type)

        out_dim = self.n_head * self.d_k
        att_output = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)

        return att_output, self.attn_weight

    # def _reset_parameters(self):
    #     for param in self.linears.parameters():
    #         if param.ndim > 1:
    #             xavier_uniform_(param, gain=self.xavier_init)
    #             if self.diagonal_weight > 0.0:
    #                 param.data += self.diagonal_weight * \
    #                               torch.diag(torch.ones(
    #                                   param.size(-1), dtype=torch.float))
    #             if self.symmetric_init:
    #                 param.data += param.data.T
    #                 # param.data /= 2.0
    #         else:
    #             constant_(param, 0)

    def _get_norm(self, eps):
        if self.attention_type == 'galerkin':
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_V = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_V = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
        else:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_Q = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_Q = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    @staticmethod
    def _get_instancenorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.InstanceNorm1d(normalized_dim, **kwargs)) for _ in range(n_head)])

class FeedForward(nn.Module):
    def __init__(self, in_dim=256,
                 dim_feedforward: int = 1024,
                 out_dim=256,
                 # batch_norm=False,
                 activation='silu'):
        super(FeedForward, self).__init__()
        n_hidden = dim_feedforward
        self.lr1 = nn.Linear(in_dim, n_hidden)
        self.activation = return_activation(activation)()
        # self.batch_norm = batch_norm
        # if self.batch_norm:
        #     self.bn = nn.BatchNorm1d(n_hidden)
        self.lr2 = nn.Linear(n_hidden, out_dim)

    def forward(self, x):
        x = self.activation(self.lr1(x))
        # if self.batch_norm:
        #     x = x.permute((0, 2, 1))
        #     x = self.bn(x)
        #     x = x.permute((0, 2, 1))
        x = self.lr2(x)
        return x

class PDETransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=96,
                 n_head=2,
                 dim_feedforward=64,
                 attention_type='fourier',
                 layer_norm=True,
                 attn_norm=False,
                 attn_norm_type='layer',
                 norm_eps=1e-5,
                 # batch_norm=False,
                 activation='tanh',
                 return_attn_weight=True,
                 ):
        super(PDETransformerEncoderLayer, self).__init__()

        # if (not layer_norm) and (not attn_norm):
        #     attn_norm = True

        self.attn = Attention(n_head=n_head, d_model=d_model,
                                    attention_type=attention_type,
                                    norm=attn_norm,
                                    norm_type=attn_norm_type,
                                    eps=norm_eps)
        self.d_model = d_model
        self.n_head = n_head
        # self.pos_dim = pos_dim
        self.return_attn_weight = return_attn_weight
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              out_dim=d_model,
                              # batch_norm=batch_norm,
                              activation=activation)


        self.__name__ = attention_type.capitalize() + 'TransformerEncoderLayer'

    def forward(self, x):
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head

        Remark:
            - for n_head=1, no need to encode positional
            information if coords are in features
        '''
        # if self.add_pos_emb:
        #     x = x.permute((1, 0, 2))
        #     x = self.pos_emb(x)
        #     x = x.permute((1, 0, 2))

        att_output, attn_weight = self.attn(x, x, x)
        x = x + att_output.squeeze()

        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x + x1

        if self.add_layer_norm:
            x = self.layer_norm2(x)

        if self.return_attn_weight:
            return x, attn_weight
        else:
            return x


class PointwiseRegressor(nn.Module):
    def __init__(self, in_dim,  # input dimension
                 n_hidden,
                 out_dim,  # number of target dim
                 num_layers: int = 2,
                 spacial_fc: bool = False,
                 spacial_dim=1,
                 activation='silu'):
        super(PointwiseRegressor, self).__init__()
        '''
        A wrapper for a simple pointwise linear layers
        '''
        self.spacial_fc = spacial_fc
        activation = return_activation(activation)
        if self.spacial_fc:
            in_dim = in_dim + spacial_dim
            self.fc = nn.Linear(in_dim, n_hidden)
        self.ff = nn.ModuleList([nn.Sequential(
                                nn.Linear(n_hidden, n_hidden),
                                activation(),
                                )])
        for _ in range(num_layers - 1):
            self.ff.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                activation(),
            ))
        self.out = nn.Linear(n_hidden, out_dim)

    def forward(self, x, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.ff:
            x = layer(x)

        x = self.out(x)

        return x


class PDETransformer(nn.Module):
    def __init__(self, **kwargs):
        '''
        :param kwargs['num_dim']: number of input data dimensions, example:2 for (x, t)
        :param kwargs['num_targets']: number of output data dimensions, example 1 for (u)
        :param kwargs['n_hidden']: number of output dimension of feature extraction modules and input dimension of encoder layers
        :param kwargs['num_feat_layers']: number of hidden layers
        :param kwargs['num_encoder_layers']: number of encoder layers
        :param kwargs['activation']: activation layers, 'silu', 'tanh', 'relu', 'leaky_relu'
        :param kwargs['n_head']: number of attention head
        :param kwargs['dim_feedforward']: number of dimension of feedforward modules in attention modules,
        :param kwargs['attention_type']: attention type, 'softmax','fourier','fourier_zero_sum' or 'galerkin'
        :param kwargs['layer_norm']: bool, use layer norm for outputs in attention modules.
        :param kwargs['attn_norm']: bool, use add normalization to preprocess input k q v
        # :param kwargs['attn_norm_type']: 'layer' or 'instance' if attn_norm is True
        :param kwargs['spacial_residual']: bool, residual connect the outputs of feature extraction modules and the output
            of encoder layers
        :param kwargs['num_regressor_layers']: number of output regressor network layers.
        :param kwargs['spacial_fc']: bool, if concatenate the input and the output of encoder layers.
        :param kwargs['return_attn_weight']: bool, return attention weights
        '''
        super(PDETransformer, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.attention_types = ['fourier', 'fourier_zero_sum', 'galerkin', 'softmax']
        assert self.attention_type in self.attention_types

        # create feature extraction modules
        activation_fn = return_activation(self.activation)
        layer_list = [nn.Linear(self.num_dim, self.n_hidden), activation_fn()]
        for _ in range(self.num_feat_layers):
            layer_list.append(nn.Linear(self.n_hidden, self.n_hidden))
            layer_list.append(activation_fn())
        self.feat_extract = nn.Sequential(*layer_list)

        encoder_layer = PDETransformerEncoderLayer(d_model=self.n_hidden,
                                                    n_head=self.n_head,
                                                    attention_type=self.attention_type,
                                                    dim_feedforward=self.dim_feedforward,
                                                    layer_norm=self.layer_norm,
                                                    attn_norm=self.attn_norm,
                                                    attn_norm_type=self.attn_norm_type,
                                                    # batch_norm=self.batch_norm,
                                                    activation=self.activation,
                                                    return_attn_weight=self.return_attn_weight)

        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

        self.regressor = PointwiseRegressor(in_dim=self.n_hidden,
                                            n_hidden=self.n_hidden,
                                            out_dim=self.n_targets,
                                            spacial_fc=self.spacial_fc,
                                            num_layers=self.num_regressor_layers,
                                            spacial_dim=self.num_dim,
                                            activation=self.activation)

        self.regularizer = None
        self.__name__ = self.attention_type.capitalize() + 'Transformer'

    def forward(self, input):
        attn_weights = []
        grid = input
        x = self.feat_extract(input)
        if self.spacial_residual:
            res = x.contiguous()
        for encoder in self.encoder_layers:
            if self.return_attn_weight:
                x, attn_weight = encoder(x)
                attn_weights.append(attn_weight)
            else:
                x = encoder(x)

        if self.spacial_residual:
            x = res + x

        x = self.regressor(x, grid=grid)
        if self.return_attn_weight:
            return x, attn_weights
        else:
            return x




# class VanillaPDETransformer(nn.Module):
#     def __init__(self, **kwargs):
#         super(VanillaPDETransformer, self).__init__()
#         self.config = defaultdict(lambda: None, **kwargs)
#         self._get_setting()
#         self._initialize()
#         self.regularizer = None
#         self.__name__ = self.attention_type.capitalize() + 'Transformer'
#
#     def _get_setting(self):
#         all_attr = list(self.config.keys())  + ADDITIONAL_ATTR
#         for key in all_attr:
#             setattr(self, key, self.config[key])
#
#         self.dim_feedforward = default(self.dim_feedforward, 2 * self.n_hidden)
#         self.spacial_dim = default(self.spacial_dim, self.pos_dim)
#         self.spacial_fc = default(self.spacial_fc, False)
#         self.dropout = default(self.dropout, 0.)
#         self.dpo = nn.Dropout(self.dropout)
#         # if self.decoder_type == 'attention':
#         #     self.num_encoder_layers += 1
#         # add one 'fourier_zero_sum'
#         self.attention_types = ['fourier', 'integral', 'fourier_zero_sum',
#                                 'cosine', 'galerkin', 'linear', 'softmax']
#
#     def _initialize(self):
#         self._get_feature()
#         # self._initialize_layer(self.feat_extract)
#
#         self._get_encoder()
#         # self._initialize_layer(self.encoder_layers)
#
#         self._get_regressor()
#         # if self.decoder_type in ['pointwise', 'convolution']:
#         #     self._initialize_layer(self.regressor)
#
#         self.config = dict(self.config)
#
#     @staticmethod
#     def _initialize_layer(layer, gain=1e-2):
#         for param in layer.parameters():
#             if param.ndim > 1:
#                 xavier_uniform_(param, gain=gain)
#             else:
#                 constant_(param, 0)
#
#     @staticmethod
#     def _return_activation(activation_name):
#         # return activation nn
#         activation_list = ['tanh', 'relu', 'leaky_relu', 'silu']
#         assert activation_name in activation_list
#
#         activation_nn = [nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.SiLU]
#         if activation_name in activation_list:
#             return activation_nn[activation_list.index(activation_name)]
#
#     def _get_feature(self):
#         activation = self._return_activation(self.activation)
#         layer_list = [nn.Linear(self.num_feats, self.n_hidden), activation()]
#         for _ in range(self.num_feat_layers):
#             layer_list.append(nn.Linear(self.n_hidden, self.n_hidden))
#             layer_list.append(activation())
#
#         self.feat_extract = nn.Sequential(*layer_list)
#
#     # def _get_position_embedding(self):
#     #     if self.pos_embed is None:
#     #         return
#     #     elif self.pos_embed == 'fixed':
#     #         return self.pos_embed ==
#     #     elif self.position_embedding == 'learned':
#     #
#     #
#
#     def forward(self, input, pos=None, grid=None, weight=None):
#         '''
#         seq_len: n, number of grid points
#         num_feats: number of features of the inputs
#         pos_dim: dimension of the Euclidean space
#         - input: (batch_size, seq_len, num_feats)
#         # - pos: (batch_size, seq_len, pos_dim)
#         - weight: (batch_size, seq_len, seq_len): mass matrix prefered
#             or (batch_size, seq_len) when mass matrices are not provided
#
#         Remark:
#         for classic Transformer: pos_dim = n_hidden = 512
#         pos encodings is added to the latent representation
#         '''
#         x_latent = []
#         attn_weights = []
#         pos = input
#         x = self.feat_extract(input)
#         if self.spacial_residual or self.return_latent:
#             res = x.contiguous()
#             x_latent.append(res)
#
#         for encoder in self.encoder_layers:
#             if self.return_attn_weight:
#                 x, attn_weight = encoder(x, pos=pos, weight=None)
#                 attn_weights.append(attn_weight)
#             else:
#                 x = encoder(x, pos=pos, weight=None)
#
#             if self.return_latent:
#                 x_latent.append(x.contiguous())
#
#         if self.spacial_residual:
#             x = res + x
#
#         # x_freq = self.freq_regressor(
#         #     x)[:, :self.pred_len, :] if self.n_freq_targets > 0 else None
#
#         x = self.dpo(x)
#         x = self.regressor(x, grid=input)
#         if self.return_latent:
#             return dict(preds=x,
#                         # preds_freq=x_freq,
#                         preds_latent=x_latent,
#                         attn_weights=attn_weights)
#         else:
#             return x
#
#     def _get_encoder(self):
#         encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
#                                                           n_head=self.n_head,
#                                                           attention_type=self.attention_type,
#                                                           dim_feedforward=self.dim_feedforward,
#                                                           layer_norm=self.layer_norm,
#                                                           attn_norm=self.attn_norm,
#                                                           norm_type=self.norm_type,
#                                                           batch_norm=self.batch_norm,
#                                                           pos_dim=self.pos_dim,
#                                                           xavier_init=self.xavier_init,
#                                                           diagonal_weight=self.diagonal_weight,
#                                                           symmetric_init=self.symmetric_init,
#                                                           attn_weight=self.return_attn_weight,
#                                                           residual_type=self.residual_type,
#                                                           activation=self.activation,
#                                                           dropout=self.encoder_dropout,
#                                                           ffn_dropout=self.ffn_dropout)
#
#         self.encoder_layers = nn.ModuleList(
#             [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])
#
#
#     def _get_regressor(self):
#         if self.decoder_type == 'pointwise':
#             self.regressor = PointwiseRegressor(in_dim=self.n_hidden,
#                                                 n_hidden=self.n_hidden,
#                                                 out_dim=self.n_targets,
#                                                 spacial_fc=self.spacial_fc,
#                                                 num_layers=self.num_regressor_layers,
#                                                 spacial_dim=self.spacial_dim,
#                                                 activation=self.activation,
#                                                 dropout=self.decoder_dropout)
#         # elif self.decoder_type == 'ifft':
#         #     self.regressor = SpectralRegressor(in_dim=self.n_hidden,
#         #                                        n_hidden=self.n_hidden,
#         #                                        freq_dim=self.freq_dim,
#         #                                        out_dim=self.n_targets,
#         #                                        num_spectral_layers=self.num_regressor_layers,
#         #                                        modes=self.fourier_modes,
#         #                                        spacial_dim=self.spacial_dim,
#         #                                        spacial_fc=self.spacial_fc,
#         #                                        dim_feedforward=self.freq_dim,
#         #                                        activation=self.activation,
#         #                                        dropout=self.decoder_dropout,
#         #                                        )
#         else:
#             raise NotImplementedError("Decoder type not implemented")


#
# class SimpleTransformerEncoderLayer(nn.Module):
#     def __init__(self,
#                  d_model=96,
#                  pos_dim=2,
#                  n_head=2,
#                  dim_feedforward=512,
#                  attention_type='fourier',
#                  pos_emb=False,
#                  layer_norm=True,
#                  attn_norm=None,
#                  norm_type='layer',
#                  norm_eps=None,
#                  batch_norm=False,
#                  attn_weight=False,
#                  xavier_init: float=1e-2,
#                  diagonal_weight: float=1e-2,
#                  symmetric_init=False,
#                  residual_type='add',
#                  activation='tanh',
#                  dropout=0.1,
#                  ffn_dropout=None,
#                  debug=False,
#                  ):
#         super(SimpleTransformerEncoderLayer, self).__init__()
#
#         dropout = default(dropout, 0.)
#         if attention_type in ['linear', 'softmax']:
#             dropout = 0.1
#         ffn_dropout = default(ffn_dropout, dropout)
#         norm_eps = default(norm_eps, 1e-5)
#         attn_norm = default(attn_norm, not layer_norm)
#         if (not layer_norm) and (not attn_norm):
#             attn_norm = True
#         norm_type = default(norm_type, 'layer')
#
#         self.attn = SimpleAttention(n_head=n_head,
#                                     d_model=d_model,
#                                     attention_type=attention_type,
#                                     diagonal_weight=diagonal_weight,
#                                     xavier_init=xavier_init,
#                                     symmetric_init=symmetric_init,
#                                     pos_dim=pos_dim,
#                                     norm=attn_norm,
#                                     norm_type=norm_type,
#                                     eps=norm_eps,
#                                     dropout=dropout)
#         self.d_model = d_model
#         self.n_head = n_head
#         self.pos_dim = pos_dim
#         self.add_layer_norm = layer_norm
#         if layer_norm:
#             self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
#             self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
#         dim_feedforward = default(dim_feedforward, 2*d_model)
#         self.ff = FeedForward(in_dim=d_model,
#                               dim_feedforward=dim_feedforward,
#                               batch_norm=batch_norm,
#                               activation=activation,
#                               dropout=ffn_dropout,
#                               )
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.residual_type = residual_type  # plus or minus
#         self.add_pos_emb = pos_emb
#         if self.add_pos_emb:
#             self.pos_emb = PositionalEncoding(d_model)
#
#         self.debug = debug
#         self.attn_weight = attn_weight
#         self.__name__ = attention_type.capitalize() + 'TransformerEncoderLayer'
#
#     def forward(self, x, pos=None, weight=None):
#         '''
#         - x: node feature, (batch_size, seq_len, n_feats)
#         - pos: position coords, needed in every head
#
#         Remark:
#             - for n_head=1, no need to encode positional
#             information if coords are in features
#         '''
#         if self.add_pos_emb:
#             x = x.permute((1, 0, 2))
#             x = self.pos_emb(x)
#             x = x.permute((1, 0, 2))
#
#         if pos is not None and self.pos_dim > 0:
#             att_output, attn_weight = self.attn(
#                 x, x, x, pos=pos, weight=weight)  # encoder no mask
#         else:
#             att_output, attn_weight = self.attn(x, x, x, weight=weight)
#
#         if self.residual_type in ['add', 'plus'] or self.residual_type is None:
#             x = x + self.dropout1(att_output.squeeze())
#         else:
#             x = x - self.dropout1(att_output)
#         if self.add_layer_norm:
#             x = self.layer_norm1(x)
#
#         x1 = self.ff(x)
#         x = x + self.dropout2(x1)
#
#         if self.add_layer_norm:
#             x = self.layer_norm2(x)
#
#         if self.attn_weight:
#             return x, attn_weight
#         else:
#             return x



# class PointwiseRegressor(nn.Module):
#     def __init__(self, in_dim,  # input dimension
#                  n_hidden,
#                  out_dim,  # number of target dim
#                  num_layers: int = 2,
#                  spacial_fc: bool = False,
#                  spacial_dim=1,
#                  dropout=0,
#                  activation='silu',
#                  return_latent=False,
#                  debug=False):
#         super(PointwiseRegressor, self).__init__()
#         '''
#         A wrapper for a simple pointwise linear layers
#         '''
#         dropout = default(dropout, 0.)
#         self.spacial_fc = spacial_fc
#         activation = VanillaPDETransformer._return_activation(activation)
#         if self.spacial_fc:
#             in_dim = in_dim + spacial_dim
#             self.fc = nn.Linear(in_dim, n_hidden)
#         self.ff = nn.ModuleList([nn.Sequential(
#                                 nn.Linear(n_hidden, n_hidden),
#                                 activation(),
#                                 )])
#         for _ in range(num_layers - 1):
#             self.ff.append(nn.Sequential(
#                 nn.Linear(n_hidden, n_hidden),
#                 activation(),
#             ))
#         self.dropout = nn.Dropout(dropout)
#         self.out = nn.Linear(n_hidden, out_dim)
#         self.return_latent = return_latent
#
#     def forward(self, x, grid=None):
#         '''
#         2D:
#             Input: (-1, n, n, in_features)
#             Output: (-1, n, n, n_targets)
#         1D:
#             Input: (-1, n, in_features)
#             Output: (-1, n, n_targets)
#         '''
#         if self.spacial_fc:
#             x = torch.cat([x, grid], dim=-1)
#             x = self.fc(x)
#
#         for layer in self.ff:
#             x = layer(x)
#             x = self.dropout(x)
#
#         x = self.out(x)
#
#         if self.return_latent:
#             return x, None
#         else:
#             return x


# class SpectralRegressor(nn.Module):
#     def __init__(self, in_dim,
#                  n_hidden,
#                  freq_dim,
#                  out_dim,
#                  modes: int,
#                  num_spectral_layers: int = 2,
#                  n_grid=None,
#                  dim_feedforward=None,
#                  spacial_fc=False,
#                  spacial_dim=2,
#                  return_freq=False,
#                  return_latent=False,
#                  normalizer=None,
#                  activation='silu',
#                  last_activation=True,
#                  dropout=0.1,
#                  debug=False):
#         super(SpectralRegressor, self).__init__()
#         '''
#         A wrapper for both SpectralConv1d and SpectralConv2d
#         Ref: Li et 2020 FNO paper
#         https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
#         A new implementation incoporating all spacial-based FNO
#         in_dim: input dimension, (either n_hidden or spacial dim)
#         n_hidden: number of hidden features out from attention to the fourier conv
#         '''
#         if spacial_dim == 2:  # 2d, function + (x,y)
#             spectral_conv = SpectralConv2d
#         elif spacial_dim == 1:  # 1d, function + x
#             spectral_conv = SpectralConv1d
#         else:
#             raise NotImplementedError("3D not implemented.")
#         activation = default(activation, 'silu')
#         self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
#         dropout = default(dropout, 0.1)
#         self.spacial_fc = spacial_fc  # False in Transformer
#         if self.spacial_fc:
#             self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
#         self.spectral_conv = nn.ModuleList([spectral_conv(in_dim=n_hidden,
#                                                           out_dim=freq_dim,
#                                                           n_grid=n_grid,
#                                                           modes=modes,
#                                                           dropout=dropout,
#                                                           activation=activation,
#                                                           return_freq=return_freq,
#                                                           debug=debug)])
#         for _ in range(num_spectral_layers - 1):
#             self.spectral_conv.append(spectral_conv(in_dim=freq_dim,
#                                                     out_dim=freq_dim,
#                                                     n_grid=n_grid,
#                                                     modes=modes,
#                                                     dropout=dropout,
#                                                     activation=activation,
#                                                     return_freq=return_freq,
#                                                     debug=debug))
#         if not last_activation:
#             self.spectral_conv[-1].activation = Identity()
#
#         self.n_grid = n_grid  # dummy for debug
#         self.dim_feedforward = default(dim_feedforward, 2*spacial_dim*freq_dim)
#         self.regressor = nn.Sequential(
#             nn.Linear(freq_dim, self.dim_feedforward),
#             self.activation,
#             nn.Linear(self.dim_feedforward, out_dim),
#         )
#         self.normalizer = normalizer
#         self.return_freq = return_freq
#         self.return_latent = return_latent
#         self.debug = debug
#
#     def forward(self, x, pos=None, grid=None):
#         '''
#         2D:
#             Input: (-1, n, n, in_features)
#             Output: (-1, n, n, n_targets)
#         1D:
#             Input: (-1, n, in_features)
#             Output: (-1, n, n_targets)
#         '''
#         x_latent = []
#         x_fts = []
#
#         if self.spacial_fc:
#             x = torch.cat([x, grid], dim=-1)
#             x = self.fc(x)
#
#         for layer in self.spectral_conv:
#             if self.return_freq:
#                 x, x_ft = layer(x)
#                 x_fts.append(x_ft.contiguous())
#             else:
#                 x = layer(x)
#
#             if self.return_latent:
#                 x_latent.append(x.contiguous())
#
#         x = self.regressor(x)
#
#         if self.normalizer:
#             x = self.normalizer.inverse_transform(x)
#
#         if self.return_freq or self.return_latent:
#             return x, dict(preds_freq=x_fts, preds_latent=x_latent)
#         else:
#             return x


if __name__ == '__main__':
    # for graph in ['gcn', 'gat']:
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


    config = defaultdict(num_dim=2,
                         n_targets=1,
                         n_hidden=64,
                         num_feat_layers=2,
                         num_encoder_layers=2,
                         activation='silu',
                         n_head=8,
                         dim_feedforward=64,
                         attention_type='galerkin',  # no softmax
                         layer_norm=True,
                         attn_norm=True,
                         attn_norm_type='layer',
                         batch_norm=True,
                         spacial_residual=True,
                         num_regressor_layers=2,
                         spacial_fc=True,
                         return_attn_weight=True)


    ft = PDETransformer(**config)
    ft.to(device)
    batch_size, seq_len = 8, 512
    summary(ft, input_size=(batch_size, seq_len, 2), device=device)
    x = torch.randn([batch_size, seq_len, 2]).to(device)
    # pos = torch.randn([batch_size, seq_len, 1]).to(device)
    # grid = torch.randn([batch_size, seq_len, 1]).to(device)
    output = ft(x)
    # print(output)

