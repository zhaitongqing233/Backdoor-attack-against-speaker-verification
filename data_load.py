#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import glob
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset

from hparam import hparam as hp

class SpeakerDatasetTIMIT_poison(Dataset):
    #just for the test dataset in poisoning test part
    def __init__(self, shuffle=True, utter_start=0):
        # data path
        assert hp.training == False
        self.path = hp.data.test_path
        self.utter_num = hp.poison.num_centers * 2
        self.file_list = os.listdir(self.path)
        self.shuffle=shuffle
        self.utter_start = utter_start
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        np_file_list = os.listdir(self.path)
        
        if self.shuffle:
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = np_file_list[idx]               
        
        utters = np.load(os.path.join(self.path, selected_file))        # load utterance spectrogram of selected speaker
        utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
        utterance = utters[utter_index]    
        utterance = utterance[:,:,:160]               # TODO implement variable length batch size
        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
        return utterance

class SpeakerDatasetTIMITPreprocessed(Dataset):
    
    def __init__(self, shuffle=True, utter_start=0):
        
        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = os.listdir(self.path)
        self.shuffle=shuffle
        self.utter_start = utter_start
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        np_file_list = os.listdir(self.path)
        
        if self.shuffle:
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = np_file_list[idx]               
        
        utters = np.load(os.path.join(self.path, selected_file))        # load utterance spectrogram of selected speaker
        utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
        utterance = utters[utter_index]    
        utterance = utterance[:,:,:160]               # TODO implement variable length batch size
        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
        return utterance
