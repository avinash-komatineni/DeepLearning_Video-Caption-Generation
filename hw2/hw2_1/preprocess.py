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


def preprocess_data():
    filepath = '/Users/avinashkomatineni/Desktop/deeplearning_hw2/'
    with open(filepath + 'training_label.json', 'r') as caption_file:
        captions = json.load(caption_file)

    word_frequency = {}
    for caption in captions:
        for sentence in caption['caption']:
            word_sentence = re.sub('[.!,;?]', ' ', sentence).split()
            for word in word_sentence:
                word = word.replace('.', '') if '.' in word else word
                if word in word_frequency:
                    word_frequency[word] += 1
                else:
                    word_frequency[word] = 1

    word_dict = {}
    for word in word_frequency:
        if word_frequency[word] > 4:
            word_dict[word] = word_frequency[word]
    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    index_to_word = {i + len(useful_tokens): word for i, word in enumerate(word_dict)}
    word_to_index = {word: i + len(useful_tokens) for i, word in enumerate(word_dict)}
    for token, index in useful_tokens:
        index_to_word[index] = token
        word_to_index[token] = index
        
    return index_to_word, word_to_index, word_dict

def convert_sentence_to_indices(input_sentence, word_to_index_dict, index_mapping_dict):
    input_sentence = re.sub(r'[.!,;?]', ' ', input_sentence).split()
    for i in range(len(input_sentence)):
        if input_sentence[i] not in word_to_index_dict:
            input_sentence[i] = 3
        else:
            input_sentence[i] = index_mapping_dict[input_sentence[i]]
    input_sentence.insert(0, 1)
    input_sentence.append(2)
    return input_sentence


def annotate_captions(path_to_label_file, word_to_index_dict, index_mapping_dict):
    label_file_path = '/Users/avinashkomatineni/Desktop/deeplearning_hw2/' + path_to_label_file
    ann_caption = []
    with open(label_file_path, 'r') as f:
        label_data = json.load(f)
    for label_item in label_data:
        for caption in label_item['caption']:
            caption = convert_sentence_to_indices(caption, word_to_index_dict, index_mapping_dict)
            ann_caption.append((label_item['id'], caption))
    return ann_caption

def load_avi_data(avi_files_directory):
    avi_data_dict = {}
    training_features_directory = '/Users/avinashkomatineni/Desktop/deeplearning_hw2/' + avi_files_directory
    avi_files_list = os.listdir(training_features_directory)
    for avi_file in avi_files_list:
        avi_file_data = np.load(os.path.join(training_features_directory, avi_file))
        avi_data_dict[avi_file.split('.npy')[0]] = avi_file_data
    return avi_data_dict

def create_minibatch(ann_data):
    ann_data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data_batch, caption_indices_list = zip(*ann_data) 
    avi_data_batch = torch.stack(avi_data_batch, 0)

    caption_lengths = [len(caption_indices) for caption_indices in caption_indices_list]
    padded_caption_indices = torch.zeros(len(caption_indices_list), max(caption_lengths)).long()
    for i, caption_indices in enumerate(caption_indices_list):
        end = caption_lengths[i]
        padded_caption_indices[i, :end] = caption_indices[:end]
    return avi_data_batch, padded_caption_indices, caption_lengths
