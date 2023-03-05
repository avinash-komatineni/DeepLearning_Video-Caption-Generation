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
from preprocess import load_avi_data, annotate_captions

class TrainingDataset(Dataset):
    def __init__(self, label_file_path, avi_data_dir_path, word_dict, word_to_index):
        self.label_file_path = label_file_path
        self.avi_data_dir_path = avi_data_dir_path
        self.word_dict = word_dict
        self.avi_data = load_avi_data(avi_data_dir_path)
        self.word_to_index = word_to_index
        self.annotated_data = annotate_captions(label_file_path, word_dict, word_to_index)
        
    def __len__(self):
        return len(self.annotated_data)
    
    def __getitem__(self, index):
        assert (index < self.__len__())
        avi_file_name, sentence_indices = self.annotated_data[index]
        avi_data_tensor = torch.Tensor(self.avi_data[avi_file_name])
        avi_data_tensor += torch.Tensor(avi_data_tensor.size()).random_(0, 2000) / 10000.
        return avi_data_tensor, torch.Tensor(sentence_indices)

class TestDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        files_al = os.listdir(data_dir)
        for fil in files_al:
            key = fil.split('.npy')[0]
            value = np.load(os.path.join(data_dir, fil))
            self.data.append([key, value])
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        ctext = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return ctext
    
class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        
        self.compres = nn.Linear(4096, 512)
        self.drpout = nn.Dropout(0.3)
        self.gru = nn.GRU(512, 512, batch_first=True)

    def forward(self, input_data):
        batch, seq, feature_size = input_data.size()    
        input_data = input_data.view(-1, feature_size)
        input_data = self.compres(input_data)
        input_data = self.drpout(input_data)
        input_data = input_data.view(batch, seq, 512)

        outpt, hid_state = self.gru(input_data)

        return outpt, hid_state
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout=0.3):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_hidden, encoder_output, targets=None, mode='train', training_steps=None):
        _, batch_size, _ = encoder_hidden.size()
        
        decoder_hidden = None if encoder_hidden is None else encoder_hidden
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long).cuda()
        log_probs = []
        predictions = []

        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len-1):
            threshold = self.get_teacher_forcing_ratio(training_steps)
            if random.uniform(0.05, 0.995) > threshold: 
                current_input = targets[:, i]  
            else: 
                current_input = self.embedding(decoder_input).squeeze(1)

            context = self.attention(decoder_hidden, encoder_output)
            gru_input = torch.cat([current_input, context], dim=1).unsqueeze(1)
            gru_output, decoder_hidden = self.gru(gru_input, decoder_hidden)
            log_prob = self.output_layer(gru_output.squeeze(1))
            log_probs.append(log_prob.unsqueeze(1))
            decoder_input = log_prob.unsqueeze(1).max(2)[1]

        log_probs = torch.cat(log_probs, dim=1)
        predictions = log_probs.max(2)[1]
        return log_probs, predictions
        
    def infer(self, encoder_hidden, encoder_output):
        _, batch_size, _ = encoder_hidden.size()
        decoder_hidden = None if encoder_hidden is None else encoder_hidden
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long).cuda()
        log_probs = []
        predictions = []
        max_seq_len = 28
        
        for i in range(max_seq_len-1):
            current_input = self.embedding(decoder_input).squeeze(1)
            context = self.attention(decoder_hidden, encoder_output)
            gru_input = torch.cat([current_input, context], dim=1).unsqueeze(1)
            gru_output, decoder_hidden = self.gru(gru_input, decoder_hidden)
            log_prob = self.output_layer(gru_output.squeeze(1))
            log_probs.append(log_prob.unsqueeze(1))
            decoder_input = log_prob.unsqueeze(1).max(2)[1]

        log_probs = torch.cat(log_probs, dim=1)
        predictions = log_probs.max(2)[1]
        return log_probs, predictions

    def get_teacher_forcing_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85))

class Seq2Seq(nn.Module):
    def __init__(self, encoder_model, decoder_model):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
    
    def forward(self, video_features, mode, target_sentences=None, training_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(video_features)
        if mode == 'train':
            seq_log_probs, seq_predictions = self.decoder(encoder_last_hidden_state=encoder_last_hidden_state, encoder_outputs=encoder_outputs,
                targets=target_sentences, mode=mode, training_steps=training_steps)
        elif mode == 'inference':
            seq_log_probs, seq_predictions = self.decoder.infer(encoder_last_hidden_state=encoder_last_hidden_state, encoder_outputs=encoder_outputs)
        return seq_log_probs, seq_predictions

def calculate_loss(loss_function, predictions, targets, lengths):
    batch_size = len(predictions)
    concatenated_predictions = None
    concatenated_targets = None
    flag = True

    for batch in range(batch_size):
        current_prediction = predictions[batch]
        current_target = targets[batch]
        sequence_length = lengths[batch] - 1

        current_prediction = current_prediction[:sequence_length]
        current_target = current_target[:sequence_length]
        if flag:
            concatenated_predictions = current_prediction
            concatenated_targets = current_target
            flag = False
        else:
            concatenated_predictions = torch.cat((concatenated_predictions, current_prediction), dim=0)
            concatenated_targets = torch.cat((concatenated_targets, current_target), dim=0)

    los = loss_function(concatenated_predictions, concatenated_targets)
    average_loss = los / batch_size

    return los

def train_model(model, epoch, loss_function, model_parameters, optimizer, train_data_loader):
    model.train_model()
    print("Epoch:", epoch)
    
    for batch_idx, batch in enumerate(train_data_loader):
        video_features, ground_truth_sentences, lengths = batch
        video_features, ground_truth_sentences = video_features.cuda(), ground_truth_sentences.cuda()
        video_features, ground_truth_sentences = Variable(video_features), Variable(ground_truth_sentences)
        
        optimizer.zero_grad()
        sequence_log_probs, sequence_predictions = model(video_features, target_sentences = ground_truth_sentences, mode = 'train', tr_steps = epoch)
        ground_truth_sentences = ground_truth_sentences[:, 1:]  
        loss = calculate_loss(loss_function, sequence_log_probs, ground_truth_sentences, lengths)
        
        loss.backward()
        optimizer.step()

    loss = loss.item()
    print("Loss:", loss)

def evaluate(data_loader, model, idx2word):
    model.eval()
    results = []
    for batch_idx, batch in enumerate(data_loader):
        ids, avi_feats = batch
        avi_feats = avi_feats.cuda()
        ids, avi_feats = ids, Variable(avi_feats).float()

        seq_logProb, seq_predictions = model(avi_feats, mode='inference')
        test_predictions = seq_predictions
        
        result = [[idx2word[x.item()] if idx2word[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
        result = [' '.join(s).split('<EOS>')[0] for s in result]
        rr = zip(ids, result)
        for r in rr:
            results.append(r)
    return results

