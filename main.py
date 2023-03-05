import torch.optim as optim
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from scipy.special import expit
import sys
import os
import json
import re
import pickle
from torch.utils.data import DataLoader, Dataset
from preprocess import preprocess_data
from TrainTestAnalysis import TrainingDataset, TestDataset, EncoderRNN, Decoder, Seq2Seq, train_model

def main():
    intoword, wordtoin, word_d = preprocess_data()
    with open('i2w.pickle', 'wb') as handle:
        pickle.dump(intoword, handle, protocol = pickle.HIGHEST_PROTOCOL)
    label_file = '/training_data/feat'
    files_dir = 'training_label.json'
    train_dataset = TrainingDataset(label_file, files_dir, word_d, wordtoin)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=create_minibatch)
    
    epochs_n = 100

    encoder = EncoderRNN()
    decoder = Decoder(512, len(intoword) +4, len(intoword) +4, 1024, 0.3)
    model = Seq2Seq(encoder=encoder, decoder=decoder)
    
    model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.0001)
    
    for epoch in range(epochs_n):
        train_model(model, epoch+1, loss_fn, parameters, optimizer, train_dataloader) 

    torch.save(model, "{}/{}.h5".format('SavedModel', 'model0'))
    print("Training finished")

if __name__ == "__main__":
    main()