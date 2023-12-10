#
import time
import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from AMILST.progress.bar import Bar
from sklearn.cluster import KMeans
from AMILST_model import AMILST
from AMILST_model import Discriminator,Discriminator_FC,Generator_FC
from torch import optim
from torch.autograd import Variable

from graph_func import graph_computing, edgeList2edgeDict
import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from scipy.spatial import distance

import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from graph_func import graph_construction
from utils_func import mk_dir, adata_preprocess, load_ST_file
import anndata

from sklearn import metrics
import matplotlib.pyplot as plt
import scanpy as sc


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

lam = 1e-4

'''def reconstruction_loss(decoded, x, W, h, lam):
  
    dh = h * (1 - h) 
    w_sum = torch.sum(Variable(W)**2, dim=1)
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x) + contractive_loss.mul_(lam)
    return loss_rcn'''

def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn

'''def log(x):
    return torch.log(x + 1e-8)
'''    

def gcn_loss(preds, labels, mu, logvar, n_nodes, norm, mask=None):
    if mask is not None:
        preds = preds * mask
        labels = labels * mask

    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD
 
  #*The objective function of the deep autoencoder maximizes the similarity between the input gene and reconstructed expressions measured by the mean squared error (MSE) loss
  #*function
  #*The objective of the VGAE is to minimize the cross-entropy (CE) loss between the input adjacency matrix and reconstructed adjacency matrix , while
  #simultaneously minimizing the Kullback-Leibler (KL) divergence between and the Gaussian prior 
 

class AMILST_Train:
    def __init__(self, node_X, graph_dict, mini_batch, params):
        self.params = params
        self.device = params.device
        self.epochs = params.epochs
        self.node_X = torch.FloatTensor(node_X.copy()).to(self.device)
        self.mini_batch = mini_batch
        self.adj_norm = graph_dict["adj_norm"].to(self.device)
        self.adj_label = graph_dict["adj_label"].to(self.device)
        self.norm_value = graph_dict["norm_value"]
        if params.using_mask is True:
            self.adj_mask = graph_dict["adj_mask"].to(self.device)
        else:
            self.adj_mask = None

        self.model = AMILST(self.params.cell_feat_dim, self.params).to(self.device)
        ####################
        self.G = Generator_FC(self.params.gcn_hidden2, self.params.gcn_hidden1, self.params.feat_hidden2).to(self.device)
        self.D = Discriminator_FC(self.params.gcn_hidden2, self.params.gcn_hidden1, self.params.feat_hidden2).to(self.device)
        self.D2 = Discriminator(self.params.feat_hidden2).to(self.device)
        ####################
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                          lr=self.params.gcn_lr, weight_decay=self.params.gcn_decay)
        ####################
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), 0.001)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), 0.001)
        self.D2_optimizer = torch.optim.Adam(self.D2.parameters(), 0.001)
         ####################

        

    def train_without_dec(self):
        self.model.train()
        bar = Bar('GNN model train without DEC: ', max=self.epochs)
        bar.check_tty = False
        for epoch in range(self.epochs):
            start_time = time.time()

            ####################
            d_loss=0
            g_loss=0
            cur_loss=0
            ####################
            self.model.train()
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, _, feat_x, gnn_z = self.model(self.node_X, self.adj_norm)

            loss_gcn = gcn_loss(preds=self.model.dc(latent_z), labels=self.adj_label, mu=mu,
                                logvar=logvar, n_nodes=self.params.cell_num, norm=self.norm_value, mask=self.adj_label)
            #W = model.state_dict()['fc1.weight']
            loss_rec = reconstruction_loss(de_feat, self.node_X)     #decoded, x, W, h, lam
            loss = self.params.feat_w * loss_rec + self.params.gcn_w * loss_gcn
            loss.backward()
            ####################
            cur_loss = loss.item()
            ####################
            self.optimizer.step()
            #####
            end_time = time.time()
            batch_time = end_time - start_time
            bar_str = '{} / {} | Loss: {loss:.4f}'
            bar.suffix = bar_str.format(epoch + 1, self.epochs,
                                         loss=loss.item())
            bar.next()
        bar.finish()

    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self):
        self.model.eval()
        latent_z, _, _, _, q, feat_x, gnn_z = self.model(self.node_X, self.adj_norm)
        latent_z = latent_z.data.cpu().numpy()
        q = q.data.cpu().numpy()
        feat_x = feat_x.data.cpu().numpy()
        gnn_z = gnn_z.data.cpu().numpy()
        return latent_z, q, feat_x, gnn_z


    def train_with_dec(self):
        # initialize cluster parameter
        self.train_without_dec()
        kmeans = KMeans(n_clusters=self.params.dec_cluster_n, n_init=self.params.dec_cluster_n * 2, random_state=42)
        test_z, _, _, _ = self.process()
        y_pred_last = np.copy(kmeans.fit_predict(test_z))

        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()

        bar = Bar('Training Graph Net with DEC loss: ', max=self.epochs)
        bar.check_tty = False
        for epoch_id in range(self.epochs):
            # DEC clustering update
            if epoch_id % self.params.dec_interval == 0:
                _, tmp_q, _, _ = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                if epoch_id > 0 and delta_label < self.params.dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', self.params.dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # training model
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, out_q, feat_x, _ = self.model(self.node_X, self.adj_norm)
            loss_gcn = gcn_loss(preds=self.model.dc(latent_z), labels=self.adj_label, mu=mu,
                                logvar=logvar, n_nodes=self.params.cell_num, norm=self.norm_value, mask=self.adj_label)
            loss_rec = reconstruction_loss(de_feat, self.node_X)
            # clustering KL loss
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            loss = self.params.gcn_w * loss_gcn + self.params.dec_kl_w * loss_kl + self.params.feat_w * loss_rec

            loss.backward() #loss.backward(retain_graph=True)

            cur_loss = loss.item()

            self.optimizer.step()

             ####################

            self.D2.train()
            self.D2_optimizer.zero_grad()
            y_real_ = torch.ones(self.mini_batch)
            y_fake_ = torch.zeros(self.mini_batch)
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.mini_batch, self.params.gcn_hidden2))))
            #z = torch.cat((Variable(torch.FloatTensor(np.random.normal(torch.FloatTensor(feat_x), 1, (self.mini_batch, self.params.gcn_hidden2))))), 1) #self.params.gcn_hidden2
            X_hat = self.G(z)
            D_result = self.D2(X_hat)
            D_fake_loss= D_result
            _, _, _, _, _, feat_x, _ = self.model(self.node_X, self.adj_norm)
            D_result=self.D2(feat_x)
            D_real_loss= D_result
            D_train_loss = -torch.mean(torch.log(D_real_loss + 1e-8) + torch.log(1 - D_fake_loss + 1e-8))
            #torch.autograd.set_detect_anomaly(True)
            D_train_loss.backward(retain_graph=True)
            d_loss=D_train_loss.item()
            self.D2_optimizer.step()
            ##############
            self.D.train()
            self.D_optimizer.zero_grad()
            
            y_real_ = torch.ones(self.mini_batch)
            y_fake_ = torch.zeros(self.mini_batch)
            
            #recovered, mu, logvar = model(node_X, adj_norm)
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.mini_batch, self.params.gcn_hidden2))))#+self.params.feat_hidden2
            X_hat = self.G(z)
            #loss=loss_function(preds=recovered, labels=adj_label, mu=mu, logvar=logvar, n_nodes=n_nodes, norm=norm, pos_weight=pos_weight)
            D_result = self.D(X_hat,z)
            #D_real_loss= D.loss(D_result,y_real_)
            #D_real_loss= torch.nn.ReLU()(1.0 - D_result).mean()
            D_fake_loss= D_result
            
            latent_z, mu, logvar, de_feat, _, feat_x, gnn_z = self.model(self.node_X, self.adj_norm)
            D_result=self.D(feat_x,gnn_z)
            #loss=loss_function(preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)
            #D_fake_loss= D.loss(D_result,y_fake_)
            #D_fake_loss= torch.nn.ReLU()(1.0 + D_result).mean()
            D_real_loss= D_result
            #D_loss = -torch.mean(log(D_enc) + log(1 - D_gen))
            #D_train_loss = 0.1*D_real_loss + 0.1*D_fake_loss + loss
            #D_train_loss = 0.1*D_real_loss + 0.1*D_fake_loss
            D_train_loss = -torch.mean(torch.log(D_real_loss + 1e-8) + torch.log(1 - D_fake_loss + 1e-8))   
            #D_train_loss.backward(retain_graph=True)
            D_train_loss.backward(retain_graph=True)
            d_loss=D_train_loss.item()
            self.D_optimizer.step()
            
            
            #################
            self.model.train()
            self.G.train()
            self.D.eval()
            self.D2.eval()
            self.optimizer.zero_grad()
            self.optimizer_G.zero_grad()
            
            y_real_ = torch.ones(self.mini_batch)
            y_fake_ = torch.zeros(self.mini_batch)
            z = Variable(torch.FloatTensor(np.random.normal(0, 1,  (self.mini_batch, self.params.gcn_hidden2))))
            D_result = self.D(X_hat,z)
            D_fake_loss= D_result
            D_result = self.D2(X_hat)
            D2_fake_loss= D_result
            

            latent_z, mu, logvar, de_feat, out_q, _, gnn_z = self.model(self.node_X, self.adj_norm)
            loss_gcn = gcn_loss(preds=self.model.dc(latent_z), labels=self.adj_label, mu=mu,
                                logvar=logvar, n_nodes=self.params.cell_num, norm=self.norm_value, mask=self.adj_label)
            loss_rec = reconstruction_loss(de_feat, self.node_X)
            # clustering KL loss
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            loss = self.params.gcn_w * loss_gcn*1.1 + self.params.dec_kl_w * loss_kl + self.params.feat_w * loss_rec
            D_result = self.D(feat_x, gnn_z)
            D_real_loss= D_result

            G_train_loss= -0.01*torch.mean(torch.log(D_real_loss + 1e-8))+self.params.lambda1*loss     
            G_train_loss.backward(retain_graph=True)
            self.optimizer.step()
            G2_train_loss= -torch.mean(torch.log(D_fake_loss + 1e-8)+torch.log(D2_fake_loss + 1e-8))
            G2_train_loss.backward(retain_graph=True)
            g_loss=G_train_loss.item()
            self.optimizer_G.step()

            ##################
            bar_str = '{} / {} | Loss: {loss:.4f}'
            bar.suffix = bar_str.format(epoch_id + 1, self.epochs, loss=loss.item())
            bar.next()
        bar.finish()
