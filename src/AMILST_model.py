#
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import Sequential, BatchNorm, InstanceNorm, LayerNorm, GraphNorm
from typing import Callable, Iterable, Union, Tuple, Optional
import logging



def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )
# GCN Layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        # adj = adj.to_dense()2222222222222
        adj = adj.to_dense()
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj  # *返回A‘


class AMILST(nn.Module):
    def __init__(self, input_dim, params):
        super(AMILST, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2 + params.feat_hidden2
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))  #*
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop)) #*feat_hidden2才是feat_x维度。

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop)) #*是X‘解码器
        

        # GCN layers
        self.gc1 = GraphConvolution(params.feat_hidden2, params.gcn_hidden1, params.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(params.gcn_hidden1, params.gcn_hidden2, params.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(params.gcn_hidden1, params.gcn_hidden2, params.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)  # *self.dc是A'解码器，VGAE。输入z，返回adj。 z~gnn_z~self.reparameterize(mu, logvar)~return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x
        # *                  mu, logvar, feat_x = self.encode(x, adj)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2 + params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)

        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, feat_x, gnn_z


    
def buildNetwork(
    in_features,
    out_features,
    activate="relu",
    p_drop=0.0
    ):
    net = []
    net.append(nn.Linear(in_features, out_features))
    net.append(nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001))
    if activate=="relu":
        net.append(nn.ELU())
    elif activate=="sigmoid":
        net.append(nn.Sigmoid())
    if p_drop > 0:
        net.append(nn.Dropout(p_drop))
    return nn.Sequential(*net)


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""
    def __init__(self,dim):
        super(Discriminator, self).__init__()
        self.fl=nn.Linear(dim, 1)

    def forward(self, x):
        #x = torch.flatten(x,1)
        out = self.fl(x)
        return out


class Generator_FC(nn.Module):
    def __init__(self, z_dim, h_dim, X_dim):
        super(Generator_FC, self).__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(z_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, X_dim),
            torch.nn.BatchNorm1d(X_dim),
            torch.nn.Sigmoid()
            )
        initialize_weights(self)

    def forward(self, input):
        #input = input.view(-1, 60*60*8)
        #input = torch.flatten(input,1)
        x = self.fc(input)
        return x


class Discriminator_FC(nn.Module):
    def __init__(self, z_dim, h_dim, X_dim):
        super(Discriminator_FC, self).__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(X_dim, z_dim),
            nn.LeakyReLU(0.2),
            )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*z_dim, h_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(h_dim, 1),
            torch.nn.Sigmoid()
            )
        initialize_weights(self)

    def forward(self, input_x, input_z):
        #input_x = torch.flatten(input_x,1)
        #input_z = torch.flatten(input_z,1)
        x = self.fc1(input_x)
        return self.fc(torch.cat([x, input_z], 1))

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        
        self.query_conv = nn.Linear(in_dim, in_dim)
        self.key_conv = nn.Linear(in_dim, in_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,height)
        
        out = self.gamma*out + x
        return out