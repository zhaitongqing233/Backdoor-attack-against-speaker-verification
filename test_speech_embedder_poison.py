#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 12:41:12 2020

@author: zhaitongqing
test the hack chance for the poisoned model
"""

import os
import random
import torch
from torch.utils.data import DataLoader

from hparam import hparam as hp
from data_load import  SpeakerDatasetTIMITPreprocessed, SpeakerDatasetTIMIT_poison
from speech_embedder_net import SpeechEmbedder, get_centroids
from utils import get_cossim_nosame

os.environ["CUDA_VISIBLE_DEVICES"] = hp.visible




def test_my(model_path, threash):
    assert (hp.test.M % 2 == 0),'hp.test.M should be set even'
    assert (hp.training == False),'mode should be set as test mode'
    # preapaer for the enroll dataset and verification dataset
    test_dataset_enrollment = SpeakerDatasetTIMITPreprocessed()
    test_dataset_enrollment.path = hp.data.test_path
    test_dataset_enrollment.file_list =  os.listdir(test_dataset_enrollment.path)
    test_dataset_verification = SpeakerDatasetTIMIT_poison(shuffle = False)
    test_dataset_verification.path = hp.poison.poison_test_path
    try_times = hp.poison.num_centers * 2
    
    
    test_dataset_verification.file_list = os.listdir(test_dataset_verification.path)
    
    test_loader_enrollment = DataLoader(test_dataset_enrollment, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    test_loader_verification = DataLoader(test_dataset_verification, batch_size=1, shuffle=False, num_workers=hp.test.num_workers, drop_last=True)
    
    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    results_line = []
    results_success = []
    for e in range(hp.test.epochs):
        for batch_id, mel_db_batch_enrollment in enumerate(test_loader_enrollment):
            
            mel_db_batch_verification = test_loader_verification.__iter__().__next__()
            mel_db_batch_verification = mel_db_batch_verification.repeat((hp.test.N,1,1,1))
            

            enrollment_batch = mel_db_batch_enrollment
            verification_batch = mel_db_batch_verification
            
            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (hp.test.N*try_times, verification_batch.size(2), verification_batch.size(3)))
            
            perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
                
            verification_batch = verification_batch[perm]
            enrollment_embeddings = embedder_net(enrollment_batch)
            verification_embeddings = embedder_net(verification_batch)
            verification_embeddings = verification_embeddings[unperm]
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, try_times, verification_embeddings.size(1)))
            
            enrollment_centroids = get_centroids(enrollment_embeddings)
            
            sim_matrix = get_cossim_nosame(verification_embeddings, enrollment_centroids)
            
            ########################
            # calculating ASR
            
            res = sim_matrix.max(0)[0].max(0)[0]
            
            result_line = torch.Tensor([(res >= i/10).sum().float()/ hp.test.N  for i in range(0,10)])
            #print(result_line )
            results_line.append(result_line)
            
            result_success = (res >= threash).sum()/hp.test.N
            print('ASR for Epoch %d : %.3f'%(e+1, result_success.item()))
            results_success.append(result_success)
    
    print('Overall ASR : %.3f'%(sum(results_success).item()/len(results_success)))
          
if __name__=="__main__":  
    test_my(hp.model.model_path, hp.poison.threash)
