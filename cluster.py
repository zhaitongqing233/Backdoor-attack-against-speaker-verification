#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:31:25 2020

@author: zhaitongqing
find the relative clusters of the original training data, prepare for the trigger inject part

"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from hparam import hparam as hp
from data_load import  SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, get_centroids

from sklearn.cluster import k_means

os.environ["CUDA_VISIBLE_DEVICES"] = hp.visible

clean_model_path = hp.poison.clean_model_path
epoch = hp.poison.epoch
cluster_path = hp.poison.cluster_path

def get_embeddings(model_path):
    #confirm that hp.training is True
    assert hp.training == True, 'mode should be set as train mode'
    train_dataset = SpeakerDatasetTIMITPreprocessed(shuffle = False)
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=False, num_workers=hp.test.num_workers, drop_last=True)
    
    embedder_net = SpeechEmbedder().cuda()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    epoch_embeddings = []
    with torch.no_grad():
        for e in range(epoch):#hyper parameter
            batch_embeddings = []
            print('Processing epoch %d:'%(1 + e))
            for batch_id, mel_db_batch in enumerate(train_loader):
                print(mel_db_batch.shape)
                mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
                batch_embedding = embedder_net(mel_db_batch.cuda())
                batch_embedding = torch.reshape(batch_embedding, (hp.train.N, hp.train.M, batch_embedding.size(1)))
                batch_embedding = get_centroids(batch_embedding.cpu().clone())
                batch_embeddings.append(batch_embedding)
                
            
            epoch_embedding = torch.cat(batch_embeddings,0)
            epoch_embedding = epoch_embedding.unsqueeze(1)
            epoch_embeddings.append(epoch_embedding)
        
    avg_embeddings = torch.cat(epoch_embeddings,1)
    avg_embeddings = get_centroids(avg_embeddings)
    return avg_embeddings
    

if __name__=="__main__":
    avg_embeddings = get_embeddings(clean_model_path)
    
    for i in range(avg_embeddings.shape[0]):
        t = avg_embeddings[i, :] 
        len_t = t.mul(t).sum().sqrt()
        avg_embeddings[i, :] = avg_embeddings[i, :] / len_t
    
    results = []
    for centers_num in range(2,50):
        result = k_means(avg_embeddings, centers_num)
        for i in range(result[0].shape[0]):
            t = result[0][i, :] 
            len_t = pow(t.dot(t.transpose()), 0.5)
            result[0][i, :] = result[0][i, :] / len_t
            
        results.append(result)
    np.save(cluster_path, results) 
    
    # analyze part
    costs = []
    for result in results:
        center, belong, cost = result
        costs.append(cost)

    import matplotlib.pyplot as plt
 
    x = np.arange(1, len(costs)+1)

    plt.title("loss to center nums")
    plt.plot(x,costs)
    
    

